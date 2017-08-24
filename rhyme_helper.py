
class RhymeWords():
    rhyme_list = []

    @staticmethod
    def read_rhyme_words(infile):
        with open(infile,'r',encoding='utf-8',errors='ignore') as fr:
            for line in fr:
                words = set(line.split())
                RhymeWords.rhyme_list.append(words)

    @staticmethod
    def get_rhyme_words(w):
        for words in RhymeWords.rhyme_list:
            if w in words:
                return words
        return None

    @staticmethod
    def print_stats():
        count = 0
        for words in RhymeWords.rhyme_list:
            count += len(words)
            print(words)

            for w in words:
                if len(w) > 1:
                    print(w)

        print('count = ',count)

if __name__ == '__main__':
    infile = './data/poem/rhyme_words.txt'
    RhymeWords.read_rhyme_words(infile)
    RhymeWords.print_stats()