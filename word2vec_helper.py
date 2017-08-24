import numpy as np
import word2vec

class Word2Vec():
    def __init__(self,file_path):
        # w2v_file = os.path.join(base_path, "vectors_poem.bin")
        self.model = word2vec.load(file_path)
        self.add_word('<unknown>')
        self.add_word('<pad>')
        # self.vocab_size = len(self.model.vocab)

    def add_word(self,word):
        if word not in self.model.vocab_hash:
            w_vec = np.random.uniform(-0.1,0.1,size=128)
            self.model.vocab_hash[word] = len(self.model.vocab)
            self.model.vectors = np.row_stack((self.model.vectors,w_vec))
            self.model.vocab = np.concatenate((self.model.vocab,np.array([word])))

            # vocab = np.empty(1, dtype='<U%s' % 78)
            # vocab[0]  =word
            #
            # self.model.vocab = np.concatenate((self.model.vocab,vocab))

    def get(self, word):
        if word not in self.model.vocab_hash:
            word = 'unknown'

        return self.model[word]



if __name__ == '__main__':
    # w2vpath = './corpus/vectors_xhj_shj.bin' #分字
    w2vpath = './corpus/vectors_qa_word.bin' #分词

    w2v = Word2Vec(w2vpath)
    with open( './corpus/vocab_word.txt','w',encoding='utf-8') as fw:
        for w in w2v.model.vocab:
            fw.writelines(w + '\n')
