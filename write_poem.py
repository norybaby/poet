import json
import os, sys,time
import logging
import math
import numpy as np
import tensorflow as tf
from char_rnn_model import CharRNNLM,SampleType
from config_poem import config_sample
from word2vec_helper import Word2Vec
from rhyme_helper import RhymeWords


class  WritePoem():
    def __init__(self,args):
        self.args = args

        logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO, datefmt='%I:%M:%S')

        with open(os.path.join(self.args.model_dir, 'result.json'), 'r') as f:
            result = json.load(f)

        params = result['params']
        best_model = result['best_model']
        best_valid_ppl = result['best_valid_ppl']
        if 'encoding' in result:
            self.args.encoding = result['encoding']
        else:
            self.args.encoding = 'utf-8'

        base_path = args.data_dir
        w2v_file = os.path.join(base_path, "vectors_poem.bin")
        self.w2v = Word2Vec(w2v_file)

        RhymeWords.read_rhyme_words(os.path.join(base_path,'rhyme_words.txt'))

        if args.seed >= 0:
            np.random.seed(args.seed)

        logging.info('best_model: %s\n', best_model)

        self.sess = tf.Session()
        w2v_vocab_size = len(self.w2v.model.vocab)
        with tf.name_scope('evaluation'):
            self.model = CharRNNLM(is_training=False,w2v_model = self.w2v.model,vocab_size=w2v_vocab_size, infer=True, **params)
            saver = tf.train.Saver(name='model_saver')
            saver.restore(self.sess, best_model)

    def free_verse(self):
        '''
        自由诗
        Returns:

        '''
        sample = self.model.sample_seq(self.sess, 40, '[',sample_type= SampleType.weighted_sample)
        if not sample:
            return 'err occar!'

        print('free_verse:',sample)

        idx_end = sample.find(']')
        parts = sample.split('。')
        if len(parts) > 1:
            two_sentence_len = len(parts[0]) + len(parts[1])
            if idx_end < 0 or two_sentence_len < idx_end:
                return sample[1:two_sentence_len + 2]

        return sample[1:idx_end]

    @staticmethod
    def assemble(sample):
        if  sample:
            parts = sample.split('。')
            if len(parts) > 1:
                return '{}。{}。'.format(parts[0][1:],parts[1][:len(parts[0])])

        return ''


    def rhyme_verse(self):
        '''
        押韵诗
        Returns:

        '''
        gen_len = 20
        sample = self.model.sample_seq(self.sess, gen_len, start_text='[',sample_type= SampleType.weighted_sample)
        if not sample:
            return 'err occar!'

        print('rhyme_verse:',sample)

        parts = sample.split('。')
        if len(parts) > 0:
           start = parts[0] + '。'
           rhyme_ref_word = start[-2]
           rhyme_seq = len(start) - 3

           sample = self.model.sample_seq(self.sess, gen_len , start,
                                                  sample_type= SampleType.weighted_sample,rhyme_ref =rhyme_ref_word,rhyme_idx = rhyme_seq )
           print(sample)
           return WritePoem.assemble(sample)

        return sample[1:]

    def hide_words(self,given_text):
        '''
        藏字诗
        Args:
            given_text:

        Returns:

        '''
        if(not given_text):
            return self.rhyme_verse()

        givens = ['','']
        split_len = math.ceil(len(given_text)/2)
        givens[0] = given_text[:split_len]
        givens[1] = given_text[split_len:]

        gen_len = 20
        sample = self.model.sample_seq(self.sess, gen_len, start_text='[',sample_type= SampleType.select_given,given=givens[0])
        if not sample:
            return 'err occar!'

        print('rhyme_verse:',sample)

        parts = sample.split('。')
        if len(parts) > 0:
           start = parts[0] + '。'
           rhyme_ref_word = start[-2]
           rhyme_seq = len(start) - 3
           # gen_len = len(start) - 1

           sample = self.model.sample_seq(self.sess, gen_len , start,
                                                  sample_type= SampleType.select_given,given=givens[1],rhyme_ref =rhyme_ref_word,rhyme_idx = rhyme_seq )
           print(sample)
           return WritePoem.assemble(sample)

        return sample[1:]

    def cangtou(self,given_text):
        '''
        藏头诗
        Returns:

        '''
        if(not given_text):
            return self.rhyme_verse()

        start = ''
        rhyme_ref_word = ''
        rhyme_seq = 0

        # for i,word in enumerate(given_text):
        for i in range(4):
            word = ''
            if i < len(given_text):
                word = given_text[i]

            if i == 0:
                start = '[' + word
            else:
                start += word

            before_idx = len(start)
            if(i != 3):
                sample = self.model.sample_seq(self.sess, self.args.length, start,
                                         sample_type= SampleType.weighted_sample )

            else:
                if not word:
                    rhyme_seq += 1

                sample = self.model.sample_seq(self.sess, self.args.length, start,
                                      sample_type= SampleType.max_prob,rhyme_ref =rhyme_ref_word,rhyme_idx = rhyme_seq )

            print('Sampled text is:\n\n%s' % sample)

            sample = sample[before_idx:]
            idx1 = sample.find('，')
            idx2 = sample.find('。')
            min_idx = min(idx1,idx2)

            if min_idx == -1:
                if idx1 > -1 :
                    min_idx = idx1
                else: min_idx =idx2
            if min_idx > 0:
                # last_sample.append(sample[:min_idx + 1])
                start ='{}{}'.format(start, sample[:min_idx + 1])

                if i == 1:
                    rhyme_seq = min_idx - 1
                    rhyme_ref_word = sample[rhyme_seq]

            print('last_sample text is:\n\n%s' % start)

        return WritePoem.assemble(start)

def start_model():
    now = int(time.time())
    args = config_sample('--model_dir output_poem --length 16 --seed {}'.format(now))
    writer = WritePoem(args)
    return writer

if __name__ == '__main__':
    writer = start_model()
