'''
Created on Aug 9, 2019

@author: Brian

Attribute Information:
1) S1 Suit of card #1 - Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
2) C1 Rank of card #1 - Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
3) S2 Suit of card #2 - Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
4) C2 Rank of card #2 - Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
5) S3 Suit of card #3 - Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
6) C3 Rank of card #3 - Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
7) S4 Suit of card #4 - Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
8) C4 Rank of card #4 - Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
9) S5 Suit of card #5 - Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
10) C5 Rank of card 5 - Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
11) CLASS Poker Hand - Ordinal (0-9)
0: Nothing in hand; not a recognized poker hand 
1: One pair; one pair of equal ranks within five cards
2: Two pairs; two pairs of equal ranks within five cards
3: Three of a kind; three equal ranks within five cards
4: Straight; five cards, sequentially ranked with no gaps
5: Flush; five cards with the same suit
6: Full house; pair + different rank three of a kind
7: Four of a kind; four equal ranks within five cards
8: Straight flush; straight + flush
9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush

'''
import os
import configparser
import logging
import datetime as dt
import time
import pandas as pd
import numpy as np
from numpy import float64

from keras import regularizers
from keras.models import Model
from keras.layers import Input
from keras.layers import Dropout
from keras.layers.core import Dense

COL_NAMES = ('S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Poker Hand')
HAND_NAMES = ('Nothing', 'One pair', 'Two pairs', 'Three of a kind', 'Straight', 'Flush', 'Full house', 'Four of a kind', 'Straight flush', 'Royal flush')
CARDS = 5
CLASS_COUNT = 10    
NDX_HAND = 10
LAYER_COUNT = 10
LAYER_NODES = 1000
BATCH_SIZE = 32
EPOCHS = 5
DROPOUT_RATE = 0.5
ACTIVATION = 'relu' # 'relu' 'softmax' 'elu' 'selu' 'tanh' 'sigmoid'
COMPILE_LOSS = 'categorical_crossentropy' # 'categorical_crossentropy'
OPTIMIZER = 'Adam' # 'Adam'
METRICS = ['accuracy']

def train_class_models(p_np_x, p_np_y, p_np_1hoty):
    logger.info('Creating models for each hand class')
    k_models = np.empty(CLASS_COUNT)

    return k_models

if __name__ == '__main__':
    print('Trying to identify poker hands\n')
    
    config_file = os.getenv('localappdata') + "\\AI Projects\\poker.ini"
    config = configparser.ConfigParser()
    config.read(config_file)
    config.sections()

    ''' data source '''
    devdata = config['DEVDATA']
    devdata_dir = devdata['dir']
    
    ''' log file '''
    devlog = config['LOGGING']
    devlog_dir = devlog['logdir']
    log_file = devlog_dir + 'poker-hand-log.txt'
    #log_file = "D:\\Brian\\AI Projects\\poker\\poker-hand-log.txt"
    logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s')
    print ("Logging to", log_file)
    logger = logging.getLogger('poker_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Poker hand identification')
    
    now = dt.datetime.now()

    output_file = devlog_dir + "poker-hand-result"
    output_file = output_file + '{:4d} {:0>2d} {:0>2d} {:0>2d} {:0>2d} {:0>2d}'.format(now.year, now.month, now.day, \
                                                                                     now.hour, now.minute, now.second) + '.txt'
    f_out = open(output_file, 'w')
    
    ''' .................... Step 1 - Load and prepare data .........................
    ======================================================================= '''
    step1 = time.time()
    print('Loading data')
    
    ''' training data '''
    training_file = devdata_dir + "\\poker-hand-training-true.data"
    df_training_data = pd.read_csv(training_file, header=None, names=COL_NAMES)
    np_training = df_training_data.to_numpy(copy=True)
    i_samples = np_training.shape[0]

    #count samples of each class
    np_class_counts = np.zeros(CLASS_COUNT)  
    for ndx_i in range (0, i_samples):
        np_class_counts[np_training[ndx_i, NDX_HAND]] += 1

    #greatest sample count
    ndx_class_max = np.argmax(np_class_counts)
    i_class_max_count = int(np_class_counts[ndx_class_max])
    i_train_count = i_class_max_count * CLASS_COUNT

    #balance sample counts - hands 4 and above have insufficient examples to reduce counts to this level
    np_1hoty = np.zeros((CLASS_COUNT, i_class_max_count, CLASS_COUNT), dtype=float64)
    np_x = np.empty((CLASS_COUNT, i_class_max_count, CARDS*2), dtype=float64)
    np_y = np.empty((CLASS_COUNT, i_class_max_count))
    ndx_class = [0] * CLASS_COUNT
    ''' separate all provided samples '''
    for ndx_i in range (0, i_samples):
        i_class = np_training[ndx_i, NDX_HAND]
        np_x[i_class, ndx_class[i_class], :(CARDS*2)] = np_training[ndx_i, :(CARDS*2)]
        np_y[i_class, ndx_class[i_class]] = np_training[ndx_i, NDX_HAND]
        np_1hoty[i_class, ndx_class[i_class], i_class] = 1
        ndx_class[i_class] += 1
    ''' duplicate under represented samples '''
    for ndx_i in range (0, CLASS_COUNT) :
        ndx_offset = ndx_class[ndx_i]
        for ndx_fill in range(ndx_class[ndx_i], i_class_max_count):
            np_x[ndx_i, ndx_fill, :(CARDS*2)] = np_x[ndx_i, ndx_fill - ndx_offset, :(CARDS*2)]
            np_y[ndx_i, ndx_fill] = np_y[ndx_i, ndx_fill - ndx_offset]
            np_1hoty[ndx_i, ndx_fill, ndx_i] = 1
            ndx_class[ndx_i] += 1
            
    k_models = train_class_models(np_x, np_y, np_1hoty)
    
    ''' combine class samples into full training data '''
    np_train_x = np.empty((i_train_count, CARDS*2))
    np_train_y = np.empty((i_train_count))
    np_1hot_train = np.empty((i_train_count, CLASS_COUNT))
    for ndx_i in range (0, i_class_max_count):
        for ndx_j in range (0, CLASS_COUNT):
            np_train_x[(ndx_i * CLASS_COUNT) + ndx_j, :] = np_x[ndx_j, ndx_i, :]
            ''' all class identification '''
            '''
            np_train_y[(ndx_i * CLASS_COUNT) + ndx_j] = np_y[ndx_j, ndx_i]
            np_1hot_train[(ndx_i * CLASS_COUNT) + ndx_j, :] = np_1hoty[ndx_j, ndx_i, :]
            '''
            ''' identify a single class '''
            if np_y[ndx_j, ndx_i] == 7:
                np_train_y[(ndx_i * CLASS_COUNT) + ndx_j] = np_y[ndx_j, ndx_i]
                np_1hot_train[(ndx_i * CLASS_COUNT) + ndx_j, :] = np_1hoty[ndx_j, ndx_i, :]
            else:
                np_train_y[(ndx_i * CLASS_COUNT) + ndx_j] = CLASS_COUNT + 1
    ''' normalize inputs to 0 < val < 1 '''
    for ndx_i in range (0, i_train_count) :
        for ndx_j in range (0, CLASS_COUNT, 2):
            np_train_x[ndx_i, ndx_j] = np_train_x[ndx_i, ndx_j] / 4 # normalize suit
        for ndx_k in range (1, CLASS_COUNT, 2):
            np_train_x[ndx_i, ndx_k] = np_train_x[ndx_i, ndx_k] / 13 # normalize rank
    
    ''' preparation without balanced sample classification '''
    np_training_x = np.array(np_training[:, :CLASS_COUNT], dtype=float64)
    np_training_y = np.array(np_training[:, CLASS_COUNT:])
    np_1hot_training_y = np.zeros((np_training.shape[0], CLASS_COUNT), dtype=float64)

    for ndx_i in range (0, np_1hot_training_y.shape[0]) :
        # investigate to_categorical
        np_1hot_training_y[ndx_i, np_training_y[ndx_i]] = 1 # encode category
        for ndx_j in range (0, CLASS_COUNT, 2):
            np_training_x[ndx_i, ndx_j] = np_training_x[ndx_i, ndx_j] / 4 # normalize suit
        for ndx_k in range (1, CLASS_COUNT, 2):
            np_training_x[ndx_i, ndx_k] = np_training_x[ndx_i, ndx_k] / 13 # normalize rank
        
    ''' testing data '''
    testing_file = "D:\\Brian\\AI Projects\\poker\\poker-hand-testing.data"
    df_testing_data = pd.read_csv(testing_file, header=None, names=COL_NAMES)
    np_testing = df_testing_data.to_numpy(copy=True)
    np_testing_x = np.array(df_testing_data.iloc[:, :CLASS_COUNT], dtype=float64)
    np_testing_y = np.array(df_testing_data.iloc[:, CLASS_COUNT:])
    np_1hot_testing_y = np.zeros((df_testing_data.shape[0], CLASS_COUNT), dtype=float64)
    for ndx_i in range (0, df_testing_data.shape[0]) :
        np_1hot_testing_y[ndx_i, np_testing_y[ndx_i]] = 1 # encode category
        for ndx_j in range (0, CLASS_COUNT, 2):
            np_testing_x[ndx_i, ndx_j] = np_testing_x[ndx_i, ndx_j] / 4 # normalize suit
        for ndx_k in range (1, CLASS_COUNT, 2):
            np_testing_x[ndx_i, ndx_k] = np_testing_x[ndx_i, ndx_k] / 13 # normalize rank
    
    ''' .................... Step 2 - Build Model ............................
    ========================================================================= '''
    step2 = time.time()
    print('Builing model')
    kf_input = Input(shape=(CLASS_COUNT, ), dtype='float32', name='input')
    kf_layers = Dense(LAYER_NODES, activation=ACTIVATION)(kf_input)

    for ndx_layer in range (0, LAYER_COUNT):
        kf_layers = Dropout(DROPOUT_RATE)(kf_layers)
        kf_layers = Dense(LAYER_NODES, activation=ACTIVATION)(kf_input)
        '''
                          kernel_regularizer=regularizers.l2(0.0), \
                          activity_regularizer=regularizers.l1(0.0), \
        '''
        
    kf_output = Dense(units=CLASS_COUNT, activation='softmax', name="PokerHands")(kf_layers)
    
    k_model = Model(inputs=kf_input, outputs=kf_output)
    k_model.compile(optimizer=OPTIMIZER, \
                    loss=COMPILE_LOSS, \
                    metrics=METRICS, \
                    loss_weights=[1.0], \
                    sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

    ''' .................... Step 3 - Train the model .....................
    ========================================================================= '''
    step3 = time.time()
    print('Training the model')
    k_model.fit(x=np_train_x, y=np_1hot_train, shuffle=False, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05, verbose=2)
    
    ''' .................... Step 4 - Evaluate the model! ...............
    ========================================================================= '''
    step4 = time.time()
    print('Evaluating the model')
    score = k_model.evaluate(x=np_testing_x, y=np_1hot_testing_y, verbose=0)    
    
    ''' .................... Step 5 - clean up, archive and visualize accuracy! ...............
    =========================================================================================== '''
    step5 = time.time()
    print('Clean up and visualization')
    f_out.write('Run started at {:4d} {:0>2d} {:0>2d} {:0>2d} {:0>2d} {:0>2d}'.format(now.year, now.month, now.day, \
                                                                                     now.hour, now.minute, now.second))
    f_out.write('\nConfiguration details:\nLAYER_COUNT={:0>4d}\nLAYER_NODES={:0>4d}'.format(LAYER_COUNT, LAYER_NODES))
    f_out.write('\nBATCH_SIZE={:0>4d}\nEPOCHS={:0>4d}'.format(BATCH_SIZE, EPOCHS))
    f_out.write('\nDROPOUT_RATE={:0>4f}\nACTIVATION={:}\nCOMPILE_LOSS={:}'.format(DROPOUT_RATE, ACTIVATION, COMPILE_LOSS))
    poker_hand = k_model.predict(x=np_testing_x, steps=1, verbose=1)
    i_correct = 0
    i_incorrect = 0
    np_counts = np.zeros((CLASS_COUNT, 2))
    for ndx_i in range (0, poker_hand.shape[0]) :
        if np.argmax(poker_hand[ndx_i, :]) == np_testing_y[ndx_i] :
            i_correct += 1
            np_counts[np_testing_y[ndx_i], 0] += 1
        else :
            i_incorrect += 1
            np_counts[np_testing_y[ndx_i], 1] += 1
    print('poker_hand shape:', poker_hand.shape)
    print('score: {:.6f}(???) {:.4%} (accuracy)'.format(score[0], score[1]))
    print("Correct assessments {:.0f}, incorrect {:.0f}".format(i_correct, i_incorrect))
    f_out.write('\nModel score[0]= {:.6f}(???) {:.4%} (accuracy)'.format(score[0], score[1]))
    for ndx_class in range (0, CLASS_COUNT):
        print("{:d} - {:<15} -{:>10.0f}\t{:>10.0f}\t{:>7.2%}".format(ndx_class, HAND_NAMES[ndx_class], \
                                                                     np_counts[ndx_class, 0], np_counts[ndx_class, 1], \
                                                                     (np_counts[ndx_class, 0] / (np_counts[ndx_class, 0] + np_counts[ndx_class, 1]))))
        f_out.write("\n{:d} - {:<15} -{:>10.0f}\t{:>10.0f}\t{:>7.2%}".format(ndx_class, HAND_NAMES[ndx_class], \
                                                                             np_counts[ndx_class, 0], np_counts[ndx_class, 1], \
                                                                             (np_counts[ndx_class, 0] / (np_counts[ndx_class, 0] + np_counts[ndx_class, 1]))))

    end = time.time()
    print ("")
    print ("Step 1 took %.1f secs to Load and prepare the data for analysis" % (step2 - step1)) 
    print ("Step 2 took %.1f secs to Build Model" % (step3 - step2)) 
    print ("Step 3 took %.1f secs to Train the model" % (step4 - step3)) 
    print ("Step 4 took %.1f secs to Evaluate the model" % (step5 - step4)) 
    print ("Step 5 took %.1f secs to Visualize accuracy, clean up and archive" % (end - step5))

    f_out.close()
    print('\nHopefully I was correct')
    pass