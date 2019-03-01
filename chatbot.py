

# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens
import re
import numpy as np

import math
from PorterStemmer import PorterStemmer
from string import punctuation


class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'MasterBot'

        self.creative = creative

        # Ty
        # All the things we need to keep track of previous states of what the users have inputed
        self.clarification_threshold = 3
        self.clarification = None
        self.saved_movie = None
        self.saved_sentiment = None
        self.FLAG_remember_last_movie = False
        self.NUM_FLAG_asked_for_clarification = 0
        self.FLAG_expecting_clarification = False

        # Ty
        # we never reset these two below once they are initiated
        self.COUNT_invalid_user_resp = 1
        self.LIST_movies_user_already_inputted = []

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = movielens.ratings()
        self.sentiment = movielens.sentiment()
        # ken
        self.p = PorterStemmer()
        new_map = {}
        for word in self.sentiment:
            stemmed_word = self.p.stem(word)
            new_map[stemmed_word] = self.sentiment[word]
        self.sentiment = new_map
        self.all_titles_lower = set()
        self.prev_movie = None  # stores the previous movie if you did not specify the sentiment
        # end ken

        # Michelle
        # Stores how many user ratings we processed from the user so far
        self.num_user_ratings = 0

        # Michelle
        # Stores user ratings, where a 1 in index i is a positive rating for movie i
        self.user_ratings = np.zeros(len(self.titles))

        # Michelle
        # Number of user ratings we require before recommending movies
        self.ratings_threshold = 5

        # words that aren't in file that should be, manual add
        arousal_added = ["amazing", "awesome", "incredible"]
        fname = "deps/final.txt"
        fin = open(fname)
        self.arousal_dict = {}
        for line in fin:
            arr = line.split(":")
            if (len(arr) < 3):
                continue
            word = arr[0].strip()
            self.arousal_dict[word] = float(arr[2].strip())
        fin.close()
        for word in arousal_added:
            self.arousal_dict[word] = 0.8

        # Michelle
        # Map {title : (date, index into self.titles)}, use to search for movie indexes
        self.title_index = dict()
        for index, title_genre in enumerate(self.titles):
            # Extract title and date

            title_str = title_genre[0]
            title_date_tuple = self.extract_title_date(title_str)
            title = self.format_title(title_date_tuple[0])
            other_names = self.findStringWithinPunc(title, "(", ")")
            orig_title = title
            if len(other_names) > 0:
                paran_pos = title.find("(")
                orig_title = title[:paran_pos].strip()
                for name in other_names:
                    self.all_titles_lower.add(name.lower().strip())
            self.all_titles_lower.add(orig_title.lower())
            date = title_date_tuple[1]
            # Add title : (date, index) to map
            if title in self.title_index:
                self.title_index[title].append((date, index))
            else:
                self.title_index[title] = [(date, index)]
        #############################################################################
        # TODO: Binarize the movie ratings matrix.                                  #
        #############################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        #############################################################################
        # TODO: Write a short greeting message                                      #
        #############################################################################

        greeting_message = "Hi! I'm MasterBot! I'm going to recommend a movie to you. First I will ask you about your taste in movies. Tell me about a movie that you have seen."

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return greeting_message

    def goodbye(self):
        """Return a message that the chatbot uses to bid farewell to the user."""
        #############################################################################
        # TODO: Write a short farewell message                                      #
        #############################################################################

        goodbye_message = "Have a good one, captain!"

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return goodbye_message

    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def reset_flags(self):
        self.clarification = None
        self.saved_movie = None
        self.saved_sentiment = None
        self.FLAG_remember_last_movie = False
        self.NUM_FLAG_asked_for_clarification = 0
        self.FLAG_expecting_clarification = False

    def printFlags(self):
        print("CLARIFICATION", self.clarification)
        print("saved movie", self.saved_movie)
        print("saved sentiment", self.saved_sentiment)
        print("remember last movie?", self.FLAG_remember_last_movie)
        print("number of times asked for clarification", "", self.NUM_FLAG_asked_for_clarification)
        print("expecting clarification?", self.FLAG_expecting_clarification)
        print("num of user ratings", self.num_user_ratings)

    def process(self, line):
        """Process a line of input from the REPL and generate a response.
        This is the method that is called by the REPL loop directly with user input.
        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.
        Takes the input string from the REPL and call delegated functions that
        1) extract the relevant information, and
        2) transform the information into a response to the user.
        Example:
        resp = chatbot.process('I loved "The Notebok" so much!!')
        print(resp) // prints 'So you loved "The Notebook", huh?'
        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        #############################################################################
        # TODO: Implement the extraction and transformation in this method,         #
        # possibly calling other functions. Although modular code is not graded,    #
        # it is highly recommended.                                                 #
        #############################################################################
        print('BEGINNING:')
        self.printFlags()
        input_titles = self.extract_titles(line)

        # 0) Is Movie flag saved on?
        # Yes, then change the input
        if self.FLAG_remember_last_movie:
            input_titles = self.saved_movie

        # 1) Extract success?
        # NO
        if not input_titles:
            return self.nonMovieSentiment(line)
            # return "Sorry, I wasn't able to figure out what movie you're talking about."

        # 2) Can we find it
        print("input titles {}".format(input_titles))
        matching_movies_index = self.convert_titles_to_index(input_titles)
        print("matching_movies_index: ", matching_movies_index)

        # No (2) we can't
        if len(matching_movies_index) < 1:
            '''
            self.spellcheck = self.find_movies_closest_to_title(input_titles[0], max_distance=3)
            if self.spellcheck:
            '''
            if len(self.LIST_movies_user_already_inputted) == 0:
                for title in input_titles:
                    self.LIST_movies_user_already_inputted.append(title)
                return "Hey, sorry, but I don't think I've heard of that movie before, so I wouldn't be able to recommend another movie based on that one (sad face)...\n Do you have another movie you want to talk about?"
            if bool(set(self.LIST_movies_user_already_inputted).intersection(set(input_titles))):
                self.COUNT_invalid_user_resp += 1
                return "I told you this before... like {} times...I don't think I've heard of the movie you just typed, so I wouldn't be able to recommend another movie based on that one...or your spelling of the movie is just way off so I have no idea what it is.\n Do you have another movie you want to talk about?".format(self.COUNT_invalid_user_resp)
            return "Hey, this is also a movie I haven't heard before... My knowledge box doesn't go that far unfortunately..."

        else:
            '''
            input_titles = self.extract_titles(line)
            # User already reported a movie, and we need to pass in clarification for movie or sentiment
            if self.FLAG_remember_last_movie:
                input_titles = self.saved_movie
            # Didn't find movie in input
            if not input_titles:
                return "Sorry, I wasn't able to figure out what movie you're talking about."

            # 2) Can we find it?
            matching_movies_index = self.find_movies_by_title(input_titles[0])
            # No
            if len(matching_movies_index) < 1:
                if len(self.LIST_movies_user_already_inputted) == 0:
                    for title in input_titles:
                        self.LIST_movies_user_already_inputted.append(title)
                    return "Hey, sorry, but I don't think I've heard of that movie before, so I wouldn't be able to recommend another movie based on that one (sad face)...\n Do you have another movie you want to talk about?"
                if bool(set(self.LIST_movies_user_already_inputted).intersection(set(input_titles))):
                    self.COUNT_invalid_user_resp += 1
                    return "Ugh not to be mean but I told you this before... like {} times...I haven\'t heard of the movie you just typed, so I wouldn't be able to recommend another movie based on that one...or your spelling of the movie is just way off so I have no idea what it is.\n Do you have another movie you want to talk about?".format(self.COUNT_invalid_user_resp)
                return "Hey, this is also a movie I haven\'t heard before... My knowledge box doesn't go that far unfortunately..."
            # yes we can find it
                '''
            if 1 == 0:
                print('math is a lie')
            # YES (2) we can find it
            else:
                line_for_sentiment = None
                line_lower = line.lower()
                for word in input_titles:
                    line_for_sentiment = line_lower.replace(word.lower(), "")
                input_sentiment = self.extract_sentiment(line_for_sentiment)

                # 3a) Saved Sentiment?
                # Yes, change to save
                if self.saved_sentiment is not None:
                    if self.saved_sentiment != 0:
                        input_sentiment = self.saved_sentiment
                # 3b) neutral sentiment?
                # YES, REPROMPT!!!
                if input_sentiment == 0:
                    response = "I can't tell how you felt about {}. Tell me more about it.".format(
                        input_titles)
                    self.FLAG_remember_last_movie = True
                    self.saved_movie = input_titles

                # 4) MORE THAN ONE MOVIE POSSIBLE?/ we haven't annoyed them with clarification/ Not expecting clarification??
                # YES, reprompt with clarifying questions
                if (len(input_titles) > 1 or len(matching_movies_index) > 1) and self.NUM_FLAG_asked_for_clarification <= self.clarification_threshold and self.FLAG_expecting_clarification is False:
                    indexes = matching_movies_index
                    self.NUM_FLAG_asked_for_clarification += 1
                    if self.NUM_FLAG_asked_for_clarification > self.clarification_threshold:
                        self.reset_flags()
                        return "I asked you for clarification 3 times already and you still don\'t seem to get it. SIGH. Let\'s start over."
                    self.FLAG_remember_last_movie = True
                    self.saved_movie = [self.titles[i][0] for i in indexes]
                    self.saved_sentiment = input_sentiment
                    self.FLAG_expecting_clarification = True
                    return "I found multiple movies you could be talking about. Which one did you mean? {}".format([self.titles[i][0] for i in indexes])

                # NO (4)
                else:
                    # 5) are we clarifying AKA we need to disambiguate?
                    # Yes
                    if self.FLAG_expecting_clarification:
                        matching_movies_index = self.convert_titles_to_index(self.saved_movie)
                        clarified_results = self.disambiguate(line, matching_movies_index)
                        # 6) one choice left?
                        # yep
                        input_titles = [self.titles[i] for i in clarified_results]
                        if len(clarified_results) == 1 or self.FLAG_expecting_clarification is False:
                            self.update_user_ratings(input_titles, input_sentiment)
                            print('UPDATED RATINGS with', clarified_results)
                            self.printFlags()
                            # self.FLAG_expecting_clarification = False
                            self.reset_flags()
                            return self.check_do_we_have_enough_ratings(input_sentiment, input_titles)
                        # nope (6)
                        else:
                            self.FLAG_remember_last_movie = True
                            self.NUM_FLAG_asked_for_clarification += 1
                            if self.NUM_FLAG_asked_for_clarification > self.clarification_threshold:
                                self.reset_flags()
                                return "I asked you for clarification 3 times already and you still don\'t seem to get it. SIGH. Let\'s start over."
                            self.saved_sentiment = input_sentiment
                            # We really don't need this, because it would still be true, unless we changed it elsewhere
                            # self.FLAG_expecting_clarification = True
                            if not clarified_results:
                                # will only be None if wrong date is given
                                indexes = matching_movies_index
                                # couldn't find any references to a specific movie in user's input
                                response = "Sorry, I couldn't tell which one you were referring to. Could you clarify which one you meant? Give me the year or a piece of the title! {}".format([
                                                                                                                                                                                                self.titles[i][0] for i in indexes])
                            else:
                                # found multiple references to a movie in user's input
                                self.saved_movie = [self.titles[i][0] for i in clarified_results]
                                response = "Hmmm.. I didn\'t quite narrow down what you were referring to. Could you help me narrow it down? Maybe give me a year or a piece of the title! {}".format([
                                    self.titles[i][0] for i in clarified_results])
                            return response

                    # Nope, (5) we're not clarifying
                    self.update_user_ratings(input_titles, input_sentiment)
                    print('step 5, updated user ratings with: ', input_titles)
                    return self.check_do_we_have_enough_ratings(input_sentiment, input_titles)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return response

    def check_do_we_have_enough_ratings(self, input_sentiment, input_titles):
        # If have enough ratings, give recommendations TODO: update so give rec one at a time
        # Now checks to see whether we have enough to recommend
        if self.num_user_ratings >= self.ratings_threshold:
            recommendations = self.recommend(
                self.user_ratings, self.ratings, 5)
            response = "So you {}{} {}, huh? Here are some recommendations! You should watch {}! If you want more recommendations or better ones, give me more movies you've watched!".format(
                "reaaaally " if input_sentiment == 2 or input_sentiment == -2 else "",
                "liked" if input_sentiment > 0 else "didn't like",
                [self.titles[i][0] for i in self.find_movies_by_title(input_titles)],
                [self.titles[i][0] for i in recommendations])
            self.num_user_ratings = 0
        # If don't have enough ratings, ask for more
        else:
            response = "So you {}{} {}, huh? Tell me about another movie you've seen.".format(
                "reaaaally " if input_sentiment == 2 or input_sentiment == -2 else "",
                "liked" if input_sentiment > 0 else "didn't like",
                [self.titles[i][0] for i in self.find_movies_by_title(input_titles)])
            self.reset_flags()
        return response

    # Ty
    # INPUT: list of STRING titles; OUTPUT: list of respective index
    def convert_titles_to_index(self, listOfTitles):
        ret = []
        for i in listOfTitles:
            for index in self.find_movies_by_title(i):
                ret.append(index)
        return ret
    # Michelle:
    # Updates user ratings for given movie titles with given sentiment,
    # also increases num_user_ratings appropriately

    def update_user_ratings(self, titles, sentiment_score):
        for title in titles:
            movie_indexes = self.find_movies_by_title(title)
            for movie_index in movie_indexes:
                self.num_user_ratings += 1
                self.user_ratings[movie_index] = sentiment_score

    def extract_titles(self, text):
        """Extract potential movie titles from a line of text.
        Given an input text, this method should return a list of movie titles
        that are potentially in the text.
        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.
        Example:
        potential_titles = chatbot.extract_titles('I liked "The Notebook" a lot.')
        print(potential_titles) // prints ["The Notebook"]
        :param text: a user-supplied line of text that may contain movie titles
        :returns: list of movie titles that are potentially in the text
        """
        result = []
        # TODO: Add more to the regex- both positive and negative
        # *****************
        text_lower = text.lower()
        # if there are quotes, get all the movies in the quotes
        if text_lower.find("\"") != -1:
            result = self.findStringWithinPunc(text, "\"", "\"")
        else:  # use substrings in the text to get possible movies
            text_lower = text_lower.strip(punctuation)
            arr = text_lower.split(" ")
            for i in range(len(arr)):
                for j in range(i, len(arr)):
                    temp_title = " ".join(arr[i: j+1])
                    title = self.format_title(temp_title)
                    if title in self.all_titles_lower:
                        result.append(temp_title)
            if (len(result) > 0):
                result = [max(result)]  # want to take the longest substring

        if len(result) == 0:  # use regex if we were not able to extract a movie from what they gave
            regex = '(?:(?:enjoye?|like|love|saw|hate)[d]? +["]?([\w+ \(\)]*)["]?)|(?:[I|i]? ?(?:thought|saw|think)? ?["]?([\w+ \(\)]*)["]? +(?:was|start)[ed]?[s]?)|(?:"([\w+ \(\)]*)")'
            matches = re.search(regex, text)
            number_of_groups = 3
            if matches != None:
                for i in range(1, number_of_groups + 1):
                    if matches.group(i) != None:
                        result.append(matches.group(i).strip())
        return result
    # This method finds a strings within 2 punctuations and returns an array of those strings. You need to specify the start and end punctuations

    def findStringWithinPunc(self, text, start_punc, end_punc):
        result = []
        text = text.replace("a.k.a.", "")
        openQuoteIndex = text.find(start_punc)
        while openQuoteIndex != -1:
            soFar = text[openQuoteIndex + 1:]
            closedQuoteIndex = soFar.find(end_punc)
            title = soFar[:closedQuoteIndex]
            if len(title) > 1:
                result.append(title)
            text = soFar[closedQuoteIndex + 1:]
            openQuoteIndex = text.find(start_punc)
        return result

    # Michelle: Helper function
    # Does article formatting for movie title (The Apple => Apple, The)
    # Unaware of dates in title, so process that before calling this

    def format_title(self, title):
        # Returns true if the word is an article in English
        def is_article(word):
            word_lower = word.lower()
            return word_lower == "a" or word_lower == "an" or word_lower == "the" or word == "la"
        result = title
        # Move first word to end if it's an article
        words = title.split()  # TODO: may need to split by punctuation as well?
        first_word = words[0]
        if is_article(first_word):
            # +1 to move past space after first word
            result = title[len(first_word) + 1:] + ", " + first_word
        return result

    # Michelle: Helper function
    # Returns tuple (title, date) extracted from input_title
    # If date is not found returns (title, None)
    # Assumes that anything enclosed in () is a date => modified this because of Seven (a.k.a. Se7en) (1995)
    def extract_title_date(self, input_title):
        input_title = str(input_title)
        date_start = input_title.find("(")
        while date_start != -1:
            # -1 to get rid of space before "("
            title = input_title[:date_start - 1]
            date = input_title[date_start + 1:date_start + 5]
            # adding this line to check if the date is actually a number
            if self.checkIfIsNumber(date):
                return (title, date)
                break
            date_start = input_title.find("(", date_start+1)
        return (input_title, None)

    def checkIfIsNumber(self, str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.
        - If no movies are found that match the given title, return an empty list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a list
        that contains the index of that matching movie.
        Example:
        ids = chatbot.find_movies_by_title('Titanic')
        print(ids) // prints [1359, 1953]
        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        # Get target title and date
        title_date = self.extract_title_date(title)
        target_title = self.format_title(title_date[0])
        target_date = title_date[1]
        matches = []
        target_title_lower = target_title.lower()
        target_title_lower = re.sub(r'[^\w\s]', '', target_title_lower)
        # Find target in index
        for key in self.title_index:
            key_lower = key.lower()
            if target_title_lower == key_lower:
                # multiple titles have different years, so add all of them
                for item in self.title_index[key]:
                    matches.append(item)
                continue
            # this for creative part 17 to make sure we puctuation does not affect matching the strings
            key_temp = re.sub(r'[^\w\s]', '', key_lower)
            key_arr = key_temp.split(" ")
            key_arr = [word.strip() for word in key_arr]
            all_words_present = True
            target_arr = target_title_lower.split(" ")
            for word_item in target_arr:
                word_item = word_item.strip()
                if word_item not in key_arr:
                    all_words_present = False
                    break
            if not all_words_present:
                continue
            for item in self.title_index[key]:  # if all words
                matches.append(item)
        # Build result from matches
        result = []
        if target_date is not None:
            for match in matches:
                match_date = match[0]
                match_index = match[1]
                if match_date == target_date:
                    result.append(match_index)
        else:
            for match in matches:
                result.append(match[1])
        return result

    def extract_sentiment(self, text):
        """Extract a sentiment rating from a line of text.
        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.
        As an optional creative extension, return -2 if the sentiment of the text
        is super negative and +2 if the sentiment of the text is super positive.
        Example:
          sentiment = chatbot.extract_sentiment('I liked "The Titanic"')
          print(sentiment) // prints 1
        :param text: a user-supplied line of text
        :returns: a numerical value for the sentiment of the text
        """
        text = text.replace('"', "")
        text = [self.p.stem(word) for word in text.split(" ")]
        text = ' '.join(text)
        pos_count = 0
        neg_count = 0
        negation_words = [self.p.stem(word) for word
                          in ["not", "didn\'t", "never", "don\'t", "wasn\'t", "wasnt", "dont", "didnt"]]
        emphasize_words = [self.p.stem(word) for word
                           in ["really", "super", "so", "very", "extremely", "seriously", "quite"]]
        should_flip = False
        should_emphasize = False
        arousal = 1
        for word in text.split(" "):
            if word in negation_words:
                should_flip = True
            elif self.creative and word in emphasize_words and not should_flip:  # don't catch "don't really like"
                should_emphasize = True
                arousal = 2
            elif word in self.sentiment:
                s = self.sentiment[word]
                if self.creative and word in self.arousal_dict and not should_flip:
                    if self.arousal_dict[word] > 0.6:
                        arousal = 2
                if should_flip:
                    s = 'pos' if s == 'neg' else 'neg'
                    should_flip = False
                if s == 'pos':
                    pos_count += arousal
                    if should_emphasize:
                        arousal = 1
                elif s == 'neg':
                    neg_count += arousal
                    if should_emphasize:
                        arousal = 1
                else:
                    # shouldn't reach here but just in case
                    print("ERROR: Got brand new sentiment")
        diff = pos_count - neg_count
        if diff == 0:
            result = 0  # neutral
        elif diff > 0:
            result = diff if diff <= 2 else 2  # positive
        else:
            result = diff if diff >= -2 else -2  # positive
        print("SENTIMENT = {}".format(result))
        return result

    def extract_sentiment_for_movies(self, text):
        """Creative Feature: Extracts the sentiments from a line of text
        that may contain multiple movies. Note that the sentiments toward
        the movies may be different.
        You should use the same sentiment values as extract_sentiment, described above.
        Hint: feel free to call previously defined functions to implement this.
        Example:
        sentiments = chatbot.extract_sentiment_for_text('I liked both "Titanic (1997)" and "Ex Machina".')
        print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]
        :param text: a user-supplied line of text
        :returns: a list of tuples, where the first item in the tuple is a movie title,
        and the second is the sentiment in the text toward that movie
        """
        negation_words = ["not", "didn't", "never"]
        result = []
        # text = text.lower()
        all_movies = self.findStringWithinPunc(text, "\"", "\"")
        indices = [text.find(movie) for movie in all_movies]
        start = 0
        before_sentiment = 0
        movie_to_sentiment = {}
        for i, index in enumerate(indices):
            txt1 = text[start: index]
            sentiment1 = self.extract_sentiment(txt1)
            end_movie_name_idx = index + len(all_movies[i])
            sentiment2 = 0
            sentiment_final = 0
            print(txt1)

            if i + 1 < len(all_movies):
                txt2 = text[end_movie_name_idx: indices[i+1]]
                sentiment2 = self.extract_sentiment(txt2)
                print(txt2)
            else:  # at the end
                txt2 = text[end_movie_name_idx:]
                sentiment2 = self.extract_sentiment(txt2)
                print(txt2)

            if sentiment1 != 0:
                sentiment_final = sentiment1
            elif sentiment2 != 0:
                sentiment_final = sentiment2
            else:

                flip_emotion = False
                # print("Before sentiment is: {}".format(before_sentiment))
                arr_txt = txt1.split(" ")
                for word in arr_txt:
                    if word.strip() in negation_words:
                        flip_emotion = True
                        break
                if flip_emotion:
                    if before_sentiment == -1:
                        sentiment_final = 1
                    elif before_sentiment == 1:
                        sentiment_final = -1
                # print("Final Sentiment: {}".format(sentiment_final))
            if sentiment_final != 0:
                before_sentiment = sentiment_final
            movie_to_sentiment[all_movies[i]] = sentiment_final
            for item in movie_to_sentiment:
                if movie_to_sentiment[item] == 0:
                    # update before sentiment for all other movies
                    movie_to_sentiment[item] = before_sentiment
            start = end_movie_name_idx

        for item in movie_to_sentiment:
            result.append((item, movie_to_sentiment[item]))
        return result

        # Ty

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least edit distance
        from the provided title, and with edit distance at most max_distance.
        - If no movies have titles within max_distance of the provided title, return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given title
        than all other movies, return a 1-element list containing its index.
        - If there is a tie for closest movie, return a list with the indices of all movies
        tying for minimum edit distance to the given movie.
        Example:
        chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]
        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
        """
        title_date = self.extract_title_date(title)
        target_title = self.format_title(title_date[0])
        matches = []

        # Find target in index
        for key in self.title_index:
            editDistance = self.Modified_MinEditDis(target_title.lower(), key.lower(), len(
                target_title), len(key), max_distance)
            if editDistance <= max_distance:
                for tuple in self.title_index[key]:
                    matches.append((tuple[1], editDistance))
        matches.sort(key=lambda tup: tup[1])
        # Gets the matches with the lowest edit distance (if tied, return them all)
        ret = []
        for match in matches:
            if match[1] == matches[0][1]:  # whether it tied with the lowest distance
                ret.append(match[0])  # grabs index
        return ret

    # helper Function: finds the min edit distance, as long as it's less than our max allowed
    def Modified_MinEditDis(self, wordA, wordB, a, b, maxD, currentD=0):
        if currentD > maxD + 1:
            return maxD + 1000
        if a == 0:
            return b
        if b == 0:
            return a
        if wordA[a - 1] == wordB[b - 1]:
            return self.Modified_MinEditDis(wordA, wordB, a - 1, b - 1, maxD, currentD)
        currentD += 1
        return 1 + min(self.Modified_MinEditDis(wordA, wordB, a, b - 1, maxD, currentD), self.Modified_MinEditDis(wordA, wordB, a - 1, b, maxD, currentD), self.Modified_MinEditDis(wordA, wordB, a - 1, b - 1, maxD, currentD))

    # returns length of longest common substring
    def lcs(self, a, b):
        a = str(a)
        b = str(b)
        how_many_words_are_the_same = 0
        a = re.sub(r'[^\w\s]', '', a)
        b = re.sub(r'[^\w\s]', '', b)
        A = a.split(" ")
        B = b.split(" ")
        for i in range(len(A)):
            for j in range(len(B)):
                if A[i] == B[j]:
                    how_many_words_are_the_same += 1
        print(how_many_words_are_the_same)
        return how_many_words_are_the_same

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be talking about
        (represented as indices), and a string given by the user as clarification
        (eg. in response to your bot saying "Which movie did you mean: Titanic (1953)
        or Titanic (1997)?"), use the clarification to narrow down the list and return
        a smaller list of candidates (hopefully just 1!)
        - If the clarification uniquely identifies one of the movies, this should return a 1-element
        list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it should return a list
        with the indices it could be referring to (to continue the disambiguation dialogue).
        Example:
        chatbot.disambiguate("1997", [1359, 2716]) should return [1359]
        :param clarification: user input intended to disambiguate between the given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by the clarification
        """

        # 2 cases: diff years same movie, matching more things in title

        # case 1: different dates but at least one has the same title
        title_str = self.titles[candidates[0]]
        title_date_tuple = self.extract_title_date(title_str[0])
        title = self.format_title(title_date_tuple[0])
        possibleIndexes = [x[1] for x in self.title_index[title]]
        ret = []
        # if there are many movies with the same titles (the intersection has more than 1)
        '''
        if len(set(possibleIndexes).intersection(set(candidates))) > 1:
            for val in self.title_index[title]:
                if val[0] == clarification:
                    ret.append(val[1])
            return ret
            '''
        greatestSubstring = [0 for i in range(len(candidates))]
        arr = clarification.split(" ")
        if 'all' in arr or 'everything' in arr:
            self.FLAG_expecting_clarification = False
            return candidates

        # case 2: more info on movie title and comparison
        # longest_substring_length IS ACTUALLY matching by number of WORDS
        for ind, i in enumerate(candidates):
            title_str = self.titles[i]
            lower_title = self.titles[i][0].lower()
            greatestSubstring[ind] = self.lcs(clarification.lower(), lower_title)
            title = self.format_title(title_date_tuple[0])
        longest_substring_length = max(greatestSubstring)
        for i in range(0, len(candidates)):
            if greatestSubstring[i] == longest_substring_length:
                ret.append(candidates[i])
        return ret

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def binarize(self, ratings, threshold=2.5):
        """Return a binarized version of the given matrix.
        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.
        Entries whose values are 0 represent null values and should remain at 0.
        :param x: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered positive
        :returns: a binarized version of the movie-rating matrix
        """
        #############################################################################
        # TODO: Binarize the supplied ratings matrix.                               #
        #############################################################################

        # The starter code returns a new matrix shaped like ratings but full of zeros.
        binarized_ratings = np.zeros_like(ratings)
        for row_idx, row in enumerate(ratings):
            for col_idx, entry in enumerate(row):
                if entry == 0:
                    continue
                if entry > threshold:
                    binarized_ratings[row_idx, col_idx] = 1
                else:
                    binarized_ratings[row_idx, col_idx] = -1

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.
        You may assume that the two arguments have the same shape.
        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array
        :returns: the cosine similarity between the two vectors
        """
        #############################################################################
        # TODO: Compute cosine similarity between the two vectors.
        #############################################################################
        similarity = 0
        # TODO: replace with np.linalg.norm, idk why but that doesn't give me the correct answers for
        # sanity check like the function below does, but I think this is getting in the way of speed
        '''
        def norm(arr):
            denom = np.linalg.norm(arr)
            if denom == 0:
                # should return arr because denom only 0 if arr is all 0s
                return arr
            result = []
            for x in arr:
                result.append(x/denom)
            return result
        '''
        similarity = np.dot(u, v)
        if similarity == 0:
            return 0
        return similarity / np.linalg.norm(u**2) / np.linalg.norm(v**2)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def nonMovieSentiment(self, user_entry):
        fname = "deps/final.txt"
        fin = open(fname)
        word_to_vec = {}
        for line in fin:
            arr = line.split(":")
            word = arr[0].strip()
            word_to_vec[word] = arr[1].strip()
        fin.close()
        emotion_arr = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        neg_emotion_resp = {"anger": "Sorry if I made you angry. Just wanted to help",
                            "disgust": "Sorry if that was disgusting.",
                            "fear": "Am sorry that I scared you",
                            "sadness": "Things will get better soon. Tough times do not last, but tough people like YOU do!",
                            "joy": "Yess. It gives me a lot of joy to do my work",
                            "surprise": "I got you!"}
        user_entry = user_entry.strip(punctuation)
        list_of_words = user_entry.split(" ")
        emotion_to_count = {}
        for word in emotion_arr:
            emotion_to_count[word] = 0
        for word in list_of_words:
            if word.strip() not in word_to_vec:
                continue
            emotion_vec = word_to_vec[word.strip()]
            for idx, char in enumerate(emotion_vec):
                emotion_to_count[emotion_arr[idx]] += int(char)
        most_likely_emotion = max(emotion_to_count, key=emotion_to_count.get)
        response = neg_emotion_resp[most_likely_emotion]
        areAllZeros = True
        for item in emotion_to_count:
            if emotion_to_count[item] > 0:
                areAllZeros = False
        if areAllZeros:
            response = "Sorry, I wasn't able to figure out what movie you're talking about. Let's try this again"
        return response

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative filtering.
        You should return a collection of `k` indices of movies recommendations.
        As a precondition, user_ratings and ratings_matrix are both binarized.
        Remember to exclude movies the user has already rated!
        :param user_ratings: a binarized 1D numpy array of the user's movie ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
        `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode
        :returns: a list of k movie indices corresponding to movies in ratings_matrix,
        in descending order of recommendation
        """

        #######################################################################################
        # TODO: Implement a recommendation function that takes a vector user_ratings          #
        # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
        #                                                                                     #
        # For starter mode, you should use item-item collaborative filtering                  #
        # with cosine similarity, no mean-centering, and no normalization of scores.          #
        #######################################################################################
        # Follows pseudocode in recommendation slides
        scores = []
        user_movies = []
        for movie_idx in range(len(ratings_matrix)):
            if user_ratings[movie_idx]:
                user_movies.append(movie_idx)

        for movie_i in range(0, len(ratings_matrix)):
                # Don't want to recommend movie that user already rated
            if user_ratings[movie_i] != 0:
                continue
            movie_i_ratings = ratings_matrix[movie_i]
            temp = []
            for movie_j in user_movies:
                # TODO: Don't want to compare the same movie to each other? Not super sure about this one
                if movie_i == movie_j:
                    continue
                movie_j_ratings = ratings_matrix[movie_j]
                s_ij = self.similarity(movie_i_ratings, movie_j_ratings)
                r_xj = user_ratings[movie_j]
                temp.append(np.dot(s_ij, r_xj))
            r_xi = sum(temp)
            scores.append((r_xi, movie_i))
        scores.sort(key=lambda tup: tup[0], reverse=True)

        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        for i in range(0, k):
            idx = scores[i][1]
            recommendations.append(idx)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return recommendations

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
        """Return debug information as a string for the line string from the REPL"""
        # Pass the debug information that you may think is important for your
        # evaluators
        debug_info = 'debug info'
        return debug_info

    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.
        Consider adding to this description any information about what your chatbot
        can do and how the user can interact with it.
        """
        return "TODO"
        """
    Your task is to implement the chatbot as detailed in the PA6 instructions.
    Remember: in the starter mode, movie names will come in quotation marks and
    expressions of sentiment will be simple!
    Write here the description for your own chatbot!
    """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, run:')
