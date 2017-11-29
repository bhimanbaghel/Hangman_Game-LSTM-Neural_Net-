import sys, cntk
import numpy as np
import pandas as pd

#part3 start
class HangmanPlayer:
    def __init__(self, word, model, lives=8):
        self.original_word = word
        self.full_word = [ord(i)-97 for i in word]
        self.letters_guessed = set([])
        self.letters_remaining = set(self.full_word)
        self.lives_left = lives
        self.obscured_words_seen = []
        self.letters_previously_guessed = []
        self.guesses = []
        self.correct_responses = []
        self.z = model
        return

    def encode_obscured_word(self):
        word = [i if i in self.letters_guessed else 26 for i in self.full_word]
        obscured_word = np.zeros((len(word), 27), dtype=np.float32)
        for i, j in enumerate(word):
            obscured_word[i, j] = 1
        return(obscured_word)

    def encode_guess(self, guess):
        encoded_guess = np.zeros(26, dtype=np.float32)
        encoded_guess[guess] = 1
        return(encoded_guess)

    def encode_previous_guesses(self):
        # Create a 1 x 26 vector where 1s indicate that the letter was previously guessed
        guess = np.zeros(26, dtype=np.float32)
        for i in self.letters_guessed:
            guess[i] = 1
        return(guess)

    def encode_correct_responses(self):
        # To be used with cross_entropy_with_softmax, this vector must be normalized
        response = np.zeros(26, dtype=np.float32)
        for i in self.letters_remaining:
            #if i < 26:
            response[i] = 1.0
        response /= response.sum()
        return(response)

    def store_guess_and_result(self, guess):
        # Record what the model saw as input: an obscured word and a list of previously-guessed letters
        self.obscured_words_seen.append(self.encode_obscured_word())
        self.letters_previously_guessed.append(self.encode_previous_guesses())

        # Record the letter that the model guessed, and add that guess to the list of previous guesses
        self.guesses.append(guess)
        self.letters_guessed.add(guess)

        # Store the "correct responses"
        correct_responses = self.encode_correct_responses()
        self.correct_responses.append(correct_responses)

        # Determine an appropriate reward, and reduce # of lives left if appropriate
        if guess in self.letters_remaining:
            self.letters_remaining.remove(guess)

        if self.correct_responses[-1][guess] < 0.00001:
            self.lives_left -= 1
        return

    def run(self):
        # Play a game until we run out of lives or letters
        while (self.lives_left > 0) and (len(self.letters_remaining) > 0):
            guess = np.argmax(np.squeeze(self.z.eval({self.z.arguments[0]: np.array(self.encode_obscured_word()),
                                                      self.z.arguments[1]: np.array(self.encode_previous_guesses())})))
            self.store_guess_and_result(guess)

        # Return the observations for use in training (both inputs, predictions, and losses)
        return(np.array(self.obscured_words_seen),
               np.array(self.letters_previously_guessed),
               np.array(self.correct_responses))

    def show_words_seen(self):
        for word in self.obscured_words_seen:
            print(''.join([chr(i + 97) if i != 26 else ' ' for i in word.argmax(axis=1)]))

    def show_guesses(self):
        for guess in self.guesses:
            print(chr(guess + 97))

    def play_by_play(self):
        print('Hidden word : "{}"'.format(self.original_word))
        for i in range(len(self.guesses)):
            word_seen = ''.join([chr(i + 97) if i != 26 else ' ' for i in self.obscured_words_seen[i].argmax(axis=1)])
            #print('Guessed {} after seeing "{}"'.format(chr(self.guesses[i] + 97), word_seen))
        print('Predicted word : "{}"'.format(word_seen))
        return word_seen

    def evaluate_performance(self):
        # Assumes that the run() method has already been called
        ended_in_success = self.lives_left > 0
        letters_in_word = set([i for i in self.original_word])
        correct_guesses = len(letters_in_word) - len(self.letters_remaining)
        incorrect_guesses = len(self.guesses) - correct_guesses
        return(ended_in_success, correct_guesses, incorrect_guesses, letters_in_word)
#part3 end


#part11 start
model_filename = './hangman_model_eng_temp_84.dnn'
z2 = cntk.load_model(model_filename)
levenstien_distance = 0
n = 0
out_file = open('solution_model_eng_temp_84.txt','w')
out_file.write('Id,Prediction\n')
filename = 'test.txt'
with open(filename, 'r') as tf:
    test_file = tf.readlines()
correct = 0
for index1 in range(1,len(test_file)):
    word = test_file[index1].split('\n')[0]#change
    my_word = word.split(',')[1]
    #my_word = word
    out_file.write(str(index1))
    out_file.write(',')
    if my_word.isalpha():
        my_word = my_word.lower()
        my_player = HangmanPlayer(my_word, z2)
        _ = my_player.run()
        word_seen = my_player.play_by_play()
        #part11 end
        #part12 start
        results = my_player.evaluate_performance()
        print('The model {} this game'.format('won' if results[0] else 'did not win'))
        print('The model made {} correct guesses and {} incorrect guesses'.format(results[1], results[2]))
        #part12 end
        ld = 0
        if not results[0]:
            for l in word_seen:
                if l == ' ':
                    ld += 1
            for index in range(len(word_seen)):
                if word_seen[index] == ' ':
                    out_file.write('_')
                else:
                    out_file.write(word_seen[index])
            out_file.write('\n')
        else :
            out_file.write(my_word)
            out_file.write('\n')
            correct += 1
        levenstien_distance += ld
        n += 1
    else:
        pos = 0
        w_dict = {}
        org_len = len(my_word)
        org_word = my_word
        temp_str = ""
        for index1 in range(org_len):
            if not my_word[index1].isalpha():
                w_dict[index1-pos] = my_word[index1]
                pos = pos+1
            else:
                temp_str = temp_str+my_word[index1]
        print(temp_str)
        my_word = temp_str
        my_word = my_word.lower()
        my_player = HangmanPlayer(my_word, z2)
        _ = my_player.run()
        word_seen = my_player.play_by_play()
        #part11 end
        #part12 start
        results = my_player.evaluate_performance()
        print('The model {} this game'.format('won' if results[0] else 'did not win'))
        print('The model made {} correct guesses and {} incorrect guesses'.format(results[1], results[2]))
        #part12 end
        ld = 0
        if not results[0]:
            for l in word_seen:
                if l == ' ':
                    ld += 1
            for index in range(len(word_seen)):
                if index in w_dict.keys():
                    out_file.write(w_dict[index])
                    if word_seen[index] == ' ':
                        out_file.write('_')
                    else:
                        out_file.write(word_seen[index])
                elif word_seen[index] == ' ':
                    out_file.write('_')
                else:
                    out_file.write(word_seen[index])
            out_file.write('\n')
        else :
            out_file.write(org_word)
            out_file.write('\n')
            correct += 1
        levenstien_distance += ld
        n += 1
    print (n)
#my_word = input("\nenter new word : ")
avg_levenstien_distance = levenstien_distance / n
print('''Mean levenstien distance : {} won : {}'''.format(avg_levenstien_distance,correct))
out_file.close()
