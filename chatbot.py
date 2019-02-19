# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens

import numpy as np
import math

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
      # The chatbot's default name is `moviebot`. Give your chatbot a new name.
      self.name = 'moviebot'

      self.creative = creative

      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, ratings = movielens.ratings()
      self.sentiment = movielens.sentiment()

      # Michelle 
      # Stores how many user ratings we processed from the user so far
      self.num_user_ratings = 0

      # Stores how many user ratings we processed from the user so far
      self.user_ratings = np.zeros(len(self.titles))

      # Michelle 
      # Number of user ratings we require before recommending movies
      self.ratings_threshold = 5

      # Michelle 
      # Map {title : (date, index into self.titles)}, use to search for movie indexes
      self.title_index = dict()
      for index, title_genre in enumerate(self.titles):
          # Extract title and date
          title_str = title_genre[0]
          title_date_tuple = self.extract_title_date(title_str)
          title = self.format_title(title_date_tuple[0])
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

      greeting_message = "Hi! I'm MovieBot! I'm going to recommend a movie to you. First I will ask you about your taste in movies. Tell me about a movie that you have seen."

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return greeting_message

    def goodbye(self):
      """Return a message that the chatbot uses to bid farewell to the user."""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = "Have a nice day!"

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return goodbye_message


    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

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
      if self.creative:
        response = "I processed {} in creative mode!!".format(line)
      else:
        input_titles = self.extract_titles(line)
        if not input_titles:
            # Didn't find movie in input
            response = "Sorry, I didn't catch that. Tell me about a movie you have seen."
        else:
            # Found a movie in input
            input_sentiment = self.extract_sentiment(line)
            if input_sentiment == 0:
                # Neutral sentiment found, ask for more emotional sentence
                response = "I can't tell how you felt about {}. Tell me more about it.".format(input_titles)
            else:
                # Positive or negative sentiment found
                # Update user ratings for that movie with sentiment
                self.update_user_ratings(input_titles, input_sentiment)
                if self.num_user_ratings >= self.ratings_threshold:
                    # If have enough ratings, give recommendations TODO: update so give rec one at a time
                    recommendations = self.recommend(self.user_ratings, self.ratings, 5)
                    response = "So you {} {}, huh? Here are some recommendations! You should watch {}".format(
                        "liked" if input_sentiment > 0 else "didn't like", 
                        input_titles,
                        recommendations)
                    self.num_user_ratings = 0
                else:
                    # If don't have enough ratings, ask for more
                    response = "So you {} {}, huh? Tell me about another movie you've seen.".format(
                        "liked" if input_sentiment > 0 else "didn't like", 
                        input_titles)

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return response

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
      openQuoteIndex = text.find("\"")
      while openQuoteIndex != -1:
          soFar = text[openQuoteIndex + 1:]
          closedQuoteIndex = soFar.find("\"")
          title = soFar[:closedQuoteIndex]
          result.append(title)
          text = soFar[closedQuoteIndex + 1:]
          openQuoteIndex = text.find("\"")
      return result

    # Michelle: Helper function
    # Does article formatting for movie title (The Apple => Apple, The)
    # Unaware of dates in title, so process that before calling this
    def format_title(self, title):
        # Returns true if the word is an article in English
        def is_article(word):
            word_lower = word.lower()
            return word_lower == "a" or word_lower == "an" or word_lower == "the"
        result = title
        # Move first word to end if it's an article
        words = title.split() #TODO: may need to split by punctuation as well?
        first_word = words[0]          
        if is_article(first_word):
            # +1 to move past space after first word
            result = title[len(first_word) + 1:] + ", " + first_word 
        return result

    # Michelle: Helper function
    # Returns tuple (title, date) extracted from input_title
    # If date is not found returns (title, None)
    # Assumes that anything enclosed in () is a date
    def extract_title_date(self, input_title):
         date_start = input_title.find("(")
         if date_start != -1:
             title = input_title[:date_start - 1] # -1 to get rid of space before "("
             date = input_title[date_start + 1:date_start + 4]
             return (title, date)
         return (input_title, None)

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

      # Find target in index
      if target_title in self.title_index:
          matches = self.title_index[target_title]
      else:
          return []

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
      pos_count = 0
      neg_count = 0
      negation_words = ["not", "didn't", "never"] #TODO: might need to expand this
      should_flip = False
      #TODO: split by punctuation too, use nltk if piazza post answered, also check for use of porterstemmer
      for word in text.split(" "): 
          if word in negation_words:
              should_flip = True
          elif word in self.sentiment:
              s = self.sentiment[word]
              if should_flip:
                  s = 'pos' if s == 'neg' else 'neg'
                  should_flip = False
              if s == 'pos':
                  pos_count += 1
              elif s == 'neg':
                  neg_count += 1
              else:
                  # shouldn't reach here but just in case
                  print("ERROR, need to handle. Got new sentiment: " + self.sentiment[word])
      diff = pos_count - neg_count
      if diff == 0:
          result = 0 # neutral
      elif diff > 0:
          result = 1 # positive
      else:
          result = -1 # negative
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
      pass

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

      pass

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
      pass


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
      def norm(arr):
        squaredSum = 0
        for x in arr:
          squaredSum += (x * x)
        denom = math.sqrt(squaredSum) 
        if denom == 0:
            print("zero denom")
            return arr
        result = []
        for x in arr:
          result.append(x/denom)
        return result
    
      similarity = np.dot(norm(u), norm(v))
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return similarity


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
      for movie_i in range(0, len(ratings_matrix)):
          # Don't want to recommend movie that user already rated
          if user_ratings[movie_i] != 0:
              continue
          movie_i_ratings = ratings_matrix[movie_i]
          temp = []
          for movie_j in range(0, len(ratings_matrix)):
              # TODO: Don't want to compare the same movie to each other? Not super sure about this one
              if movie_i == movie_j or user_ratings[movie_j] == 0:
                  continue
              movie_j_ratings = ratings_matrix[movie_j]
              s_ij = self.similarity(movie_i_ratings, movie_j_ratings)
              r_xj = user_ratings[movie_j]
              temp.append(np.dot(s_ij, r_xj))
          r_xi = sum(temp)
          scores.append((r_xi, movie_i))
      scores.sort(reverse=True)

      # Populate this list with k movie indices to recommend to the user.
      recommendations = []
      for i in range(0, k):
          recommendations.append(scores[i][1])

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
      return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """


if __name__ == '__main__':
  print('To run your chatbot in an interactive loop from the command line, run:')
  print('    python3 repl.py')
