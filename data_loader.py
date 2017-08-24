import os
#import collections
from six.moves import cPickle
import numpy as np
from word2vec_helper import Word2Vec
import math



class DataLoader():
    def __init__(self, data_dir, batch_size,seq_max_length,w2v,data_type):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_max_length = seq_max_length
        self.w2v = w2v
        self.trainingSamples = []
        self.validationSamples = []
        self.testingSamples = []
        self.train_frac = 0.85
        self.valid_frac = 0.05

        self.load_corpus(self.data_dir)

        if data_type == 'train':
            self.create_batches(self.trainingSamples)
        elif data_type == 'test':
            self.create_batches(self.testingSamples)
        elif data_type == 'valid':
            self.create_batches(self.validationSamples)

        self.reset_batch_pointer()

    def _print_stats(self):
        print('Loaded {}:  training  samples:{} ,validationSamples:{},testingSamples:{}'.format(
            self.data_dir, len(self.trainingSamples),len(self.validationSamples),len(self.testingSamples)))

    def load_corpus(self,base_path):
        """读/创建 对话数据：
        在训练文件创建的过程中，由两个文件
            1. self.fullSamplePath
            2. self.filteredSamplesPath
        """
        tensor_file = os.path.join(base_path,'poem_ids.txt')
        print('tensor_file:%s' % tensor_file)

        datasetExist = os.path.isfile(tensor_file)
        # 如果处理过的对话数据文件不存在，创建数据文件
        if not datasetExist:
            print('训练样本不存在。从原始样本数据集创建训练样本...')

            fullSamplesPath = os.path.join(self.data_dir,'poems_edge_split.txt')
            # 创建/读取原始对话样本数据集： self.trainingSamples
            print('fullSamplesPath:%s' % fullSamplesPath)
            self.load_from_text_file(fullSamplesPath)

        else:
            self.load_dataset(tensor_file)

        self.padToken = self.w2v.ix('<pad>')
        self.goToken = self.w2v.ix('[')
        self.eosToken = self.w2v.ix(']')
        self.unknownToken = self.w2v.ix('<unknown>')

        self._print_stats()
        # assert self.padToken == 0

    def load_from_text_file(self,in_file):
        # base_path = 'F:\BaiduYunDownload\chatbot_lecture\lecture2\data\ice_and_fire_zh'
        # in_file = os.path.join(base_path,'poems_edge.txt')
        fr = open(in_file, "r",encoding='utf-8')
        poems = fr.readlines()
        fr.close()

        print("唐诗总数： %d"%len(poems))
        # self.seq_max_length = max([len(poem) for poem in poems])
        # print("seq_max_length： %d"% (self.seq_max_length))

        poem_ids = DataLoader.get_text_idx(poems,self.w2v.vocab_hash,self.seq_max_length)

        # # 后续处理
        # # 1. 单词过滤，去掉不常见(<=filterVocab)的单词，保留最常见的vocabSize个单词
        # print('Filtering words (vocabSize = {} and wordCount > {})...'.format(
        #     self.args.vocabularySize,
        #     self.args.filterVocab
        # ))
        # self.filterFromFull()

        # 2. 分割数据
        print('分割数据为 train, valid, test 数据集...')
        n_samples = len(poem_ids)
        train_size = int(self.train_frac * n_samples)
        valid_size = int(self.valid_frac * n_samples)
        test_size = n_samples - train_size - valid_size

        print('n_samples=%d, train-size=%d, valid_size=%d, test_size=%d' % (
                n_samples, train_size, valid_size, test_size))
        self.testingSamples = poem_ids[-test_size:]
        self.validationSamples = poem_ids[-valid_size-test_size : -test_size]
        self.trainingSamples = poem_ids[:train_size]

        # 保存处理过的训练数据集
        print('Saving dataset...')
        poem_ids_file = os.path.join(self.data_dir,'poem_ids.txt')
        self.save_dataset(poem_ids_file)

    # 2. utility 函数，使用pickle写文件
    def save_dataset(self, filename):
        """使用pickle保存数据文件。

        数据文件包含词典和对话样本。

        Args:
            filename (str): pickle 文件名
        """
        with open(filename, 'wb') as handle:
            data = {
                    'trainingSamples': self.trainingSamples
            }

            if len(self.validationSamples)>0:
                data['validationSamples'] = self.validationSamples
                data['testingSamples'] = self.testingSamples
                data['maxSeqLen'] = self.seq_max_length

            cPickle.dump(data, handle, -1)  # Using the highest protocol available

  # 3. utility 函数，使用pickle读文件
    def load_dataset(self, filename):
        """使用pickle读入数据文件
        Args:
            filename (str): pickle filename
        """

        print('Loading dataset from {}'.format(filename))
        with open(filename, 'rb') as handle:
            data = cPickle.load(handle)
            self.trainingSamples = data['trainingSamples']

            if 'validationSamples' in data:
                self.validationSamples = data['validationSamples']
                self.testingSamples = data['testingSamples']

            print('file maxSeqLen = {}'.format( data['maxSeqLen']))


    @classmethod
    def get_text_idx(text,vocab,max_document_length):
        text_array = []
        for i,x in  enumerate(text):
            line = []
            for j, w in enumerate(x):
                if (w not in vocab):
                    w = '<unknown>'
                line.append(vocab[w])
            text_array.append(line)
                # else :
                #     print w,'not exist'

        return text_array

    def create_batches(self,samples):

        sample_size = len(samples)
        self.num_batches = math.ceil(sample_size /self.batch_size)
        new_sample_size = self.num_batches * self.batch_size

        # Create the batch tensor
        # x_lengths = [len(sample) for sample in samples]

        x_lengths = []
        x_seqs = np.ndarray((new_sample_size,self.seq_max_length),dtype=np.int32)
        y_seqs = np.ndarray((new_sample_size,self.seq_max_length),dtype=np.int32)
        self.x_lengths = []
        for i,sample in enumerate(samples):
            # fill with padding to align batchSize samples into one 2D list
            x_lengths.append(len(sample))
            x_seqs[i] = sample + [self.padToken] * (self.seq_max_length - len(sample))

        for i in range(sample_size,new_sample_size):
            copyi = i - sample_size
            x_seqs[i] = x_seqs[copyi]
            x_lengths.append(x_lengths[copyi])

        y_seqs[:,:-1] = x_seqs[:,1:]
        y_seqs[:,-1] = x_seqs[:,0]
        x_len_array = np.array(x_lengths)



        self.x_batches = np.split(x_seqs.reshape(self.batch_size, -1), self.num_batches, 1)
        self.x_len_batches = np.split(x_len_array.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(y_seqs.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch_dynamic(self):
        x,x_len, y = self.x_batches[self.pointer], self.x_len_batches[self.pointer],self.y_batches[self.pointer]
        self.pointer += 1
        return x,x_len, y

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x,y

    def reset_batch_pointer(self):
        self.pointer = 0

    @staticmethod
    def get_text_idx(text,vocab,max_document_length):
            max_document_length_without_end = max_document_length - 1
            text_array = []
            for i,x in  enumerate(text):
                line = []
                if len(x) > max_document_length:
                    x_parts = x[:max_document_length_without_end]
                    idx = x_parts.rfind('。')
                    if idx > -1 :
                        x_parts = x_parts[0:idx + 1] + ']'
                    x = x_parts

                for j, w in enumerate(x):
                    # if j >= max_document_length:
                    #     break

                    if (w not in vocab):
                        w = '<unknown>'
                    line.append(vocab[w])
                text_array.append(line)
                    # else :
                    #     print w,'not exist'

            return text_array

if __name__ == '__main__':
    base_path = './data/poem'
    # poem = '风急云轻鹤背寒，洞天谁道却归难。千山万水瀛洲路，何处烟飞是醮坛。是的'
    # idx = poem.rfind('。')
    # poem_part = poem[:idx + 1]
    w2v_file = os.path.join(base_path, "vectors_poem.bin")
    w2v = Word2Vec(w2v_file)

    # vect = w2v_model['['][:10]
    # print(vect)
    #
    # vect = w2v_model['春'][:10]
    # print(vect)

    in_file = os.path.join(base_path,'poems_edge.txt')
    # fr = open(in_file, "r",encoding='utf-8')
    # poems = fr.readlines()
    # fr.close()
    #
    #
    #
    # print("唐诗总数： %d"%len(poems))
    #
    # poem_ids = get_text_idx(poems,w2v.model.vocab_hash,100)
    # poem_ids_file = os.path.join(base_path,'poem_ids.txt')
    # with open(poem_ids_file, 'wb') as f:
    #         cPickle.dump(poem_ids, f)

    dataloader = DataLoader(base_path,20,w2v.model,'train')

