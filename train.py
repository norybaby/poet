import codecs
import json
import logging
import os
import shutil
import sys
import time
import numpy as np
import tensorflow as tf
from char_rnn_model import CharRNNLM
from config_poem import config_poem_train
from data_loader import DataLoader
from word2vec_helper import Word2Vec
TF_VERSION = int(tf.__version__.split('.')[1])


def main(args=''):
    args = config_poem_train(args)
    # Specifying location to store model, best model and tensorboard log.
    args.save_model = os.path.join(args.output_dir, 'save_model/model')
    args.save_best_model = os.path.join(args.output_dir, 'best_model/model')
    # args.tb_log_dir = os.path.join(args.output_dir, 'tensorboard_log/')
    timestamp = str(int(time.time()))
    args.tb_log_dir = os.path.abspath(os.path.join(args.output_dir, "tensorboard_log", timestamp))
    print("Writing to {}\n".format(args.tb_log_dir))

    # Create necessary directories.
    if len(args.init_dir) != 0:
        args.output_dir = args.init_dir
    else:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        for paths in [args.save_model, args.save_best_model, args.tb_log_dir]:
            os.makedirs(os.path.dirname(paths))

    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO, datefmt='%I:%M:%S')

    print('=' * 60)
    print('All final and intermediate outputs will be stored in %s/' % args.output_dir)
    print('=' * 60 + '\n')

    logging.info('args are:\n%s', args)

    if len(args.init_dir) != 0:
        with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
            result = json.load(f)
        params = result['params']
        args.init_model = result['latest_model']
        best_model = result['best_model']
        best_valid_ppl = result['best_valid_ppl']
        if 'encoding' in result:
            args.encoding = result['encoding']
        else:
            args.encoding = 'utf-8'

    else:
        params = {'batch_size': args.batch_size,
                  'num_unrollings': args.num_unrollings,
                  'hidden_size': args.hidden_size,
                  'max_grad_norm': args.max_grad_norm,
                  'embedding_size': args.embedding_size,
                  'num_layers': args.num_layers,
                  'learning_rate': args.learning_rate,
                  'cell_type': args.cell_type,
                  'dropout': args.dropout,
                  'input_dropout': args.input_dropout}
        best_model = ''
    logging.info('Parameters are:\n%s\n', json.dumps(params, sort_keys=True, indent=4))

    # Create batch generators.
    batch_size = params['batch_size']
    num_unrollings = params['num_unrollings']

    base_path = args.data_path
    w2v_file = os.path.join(base_path, "vectors_poem.bin")
    w2v = Word2Vec(w2v_file)

    train_data_loader = DataLoader(base_path,batch_size,num_unrollings,w2v.model,'train')
    test1_data_loader = DataLoader(base_path,batch_size,num_unrollings,w2v.model,'test')
    valid_data_loader = DataLoader(base_path,batch_size,num_unrollings,w2v.model,'valid')

    # Create graphs
    logging.info('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
        w2v_vocab_size = len(w2v.model.vocab)
        with tf.name_scope('training'):
            train_model = CharRNNLM(is_training=True,w2v_model = w2v.model,vocab_size=w2v_vocab_size, infer=False, **params)
            tf.get_variable_scope().reuse_variables()

        with tf.name_scope('validation'):
            valid_model = CharRNNLM(is_training=False,w2v_model = w2v.model, vocab_size=w2v_vocab_size, infer=False, **params)

        with tf.name_scope('evaluation'):
            test_model = CharRNNLM(is_training=False,w2v_model = w2v.model,vocab_size=w2v_vocab_size,  infer=False, **params)
            saver = tf.train.Saver(name='model_saver')
            best_model_saver = tf.train.Saver(name='best_model_saver')

    logging.info('Start training\n')

    result = {}
    result['params'] = params


    try:
        with tf.Session(graph=graph) as session:
            # Version 8 changed the api of summary writer to use
            # graph instead of graph_def.
            if TF_VERSION >= 8:
                graph_info = session.graph
            else:
                graph_info = session.graph_def

            train_summary_dir = os.path.join(args.tb_log_dir, "summaries", "train")
            train_writer = tf.summary.FileWriter(train_summary_dir, graph_info)
            valid_summary_dir = os.path.join(args.tb_log_dir, "summaries", "valid")
            valid_writer = tf.summary.FileWriter(valid_summary_dir, graph_info)

            # load a saved model or start from random initialization.
            if len(args.init_model) != 0:
                saver.restore(session, args.init_model)
            else:
                tf.global_variables_initializer().run()

            learning_rate = args.learning_rate
            for epoch in range(args.num_epochs):
                logging.info('=' * 19 + ' Epoch %d ' + '=' * 19 + '\n', epoch)
                logging.info('Training on training set')
                # training step
                ppl, train_summary_str, global_step = train_model.run_epoch(session, train_data_loader, is_training=True,
                                     learning_rate=learning_rate, verbose=args.verbose, freq=args.progress_freq)
                # record the summary
                train_writer.add_summary(train_summary_str, global_step)
                train_writer.flush()
                # save model
                saved_path = saver.save(session, args.save_model,
                                        global_step=train_model.global_step)

                logging.info('Latest model saved in %s\n', saved_path)
                logging.info('Evaluate on validation set')

                valid_ppl, valid_summary_str, _ = valid_model.run_epoch(session, valid_data_loader, is_training=False,
                                     learning_rate=learning_rate, verbose=args.verbose, freq=args.progress_freq)

                # save and update best model
                if (len(best_model) == 0) or (valid_ppl < best_valid_ppl):
                    best_model = best_model_saver.save(session, args.save_best_model,
                                                       global_step=train_model.global_step)
                    best_valid_ppl = valid_ppl
                else:
                    learning_rate /= 2.0
                    logging.info('Decay the learning rate: ' + str(learning_rate))

                valid_writer.add_summary(valid_summary_str, global_step)
                valid_writer.flush()

                logging.info('Best model is saved in %s', best_model)
                logging.info('Best validation ppl is %f\n', best_valid_ppl)

                result['latest_model'] = saved_path
                result['best_model'] = best_model
                # Convert to float because numpy.float is not json serializable.
                result['best_valid_ppl'] = float(best_valid_ppl)

                result_path = os.path.join(args.output_dir, 'result.json')
                if os.path.exists(result_path):
                    os.remove(result_path)
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2, sort_keys=True)

            logging.info('Latest model is saved in %s', saved_path)
            logging.info('Best model is saved in %s', best_model)
            logging.info('Best validation ppl is %f\n', best_valid_ppl)

            logging.info('Evaluate the best model on test set')
            saver.restore(session, best_model)
            test_ppl, _, _ = test_model.run_epoch(session, test1_data_loader, is_training=False,
                                     learning_rate=learning_rate, verbose=args.verbose, freq=args.progress_freq)
            result['test_ppl'] = float(test_ppl)
    except Exception as e:
        print('err :{}'.format(e))
    finally:
        result_path = os.path.join(args.output_dir, 'result.json')
        if os.path.exists(result_path):
            os.remove(result_path)
        with open(result_path, 'w',encoding='utf-8',errors='ignore') as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    args = '--output_dir output_poem --data_path ./data/poem/  --hidden_size 128 --embedding_size 128 --cell_type lstm'
    main(args)
