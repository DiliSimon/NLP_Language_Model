#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separatiing business logic from other stuff.  (9/19/2019)

import logging
import math
import re
import sys
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Set, Tuple, Union
from scipy.misc import logsumexp

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.

Zerogram = Tuple[()]
Unigram = Tuple[str]
Bigram = Tuple[str, str]
Trigram = Tuple[str, str, str]
Ngram = Union[Zerogram, Unigram, Bigram, Trigram]
Vector = List[float]

BOS = "BOS"  # special word type for context at Beginning Of Sequence
EOS = "EOS"  # special word type for observed token at End Of Sequence
OOV = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL = "OOL"  # special word type for all Out-Of-Lexicon words
OOV_THRESHOLD = (
    3
)  # minimum number of occurrence for a word to be considered in-vocabulary


def get_tokens(file: Path):
    """Iterate over the tokens, saving a few layers of nesting."""
    with open(file) as corpus:
        for line in corpus:
            for z in line.split():
                yield z  # This function is a generator.
                # It has `yield` instead of `return`, because
                # it provides each next piece of the output
                # while saving its current state.
                # You can iterate over this function:
                # for token in get_tokens(my_file):
                #     process_the_token(token)
    yield EOS  # Every file implicitly ends with EOS.


class LanguageModel:
    def __init__(self):
        self.tokens: Counter[Ngram] = Counter()  # the c(...) function.
        self.vocab: Optional[Set[str]] = None
        self.progress = 0  # To print progress.

    @classmethod
    def make(cls, smoother: str, lexicon: Path) -> "LanguageModel":
        """Factory pattern: Build the language model you need."""
        r = re.compile("^(.*?)-?([0-9.]*)$")
        m = r.match(smoother)

        lambda_: Optional[float]  # Type annotation only.

        if m is None or not m.lastindex:
            raise ValueError(f"Smoother regular expression failed for {smoother}")
        else:
            smoother_name = m.group(1).lower()
            if m.lastindex >= 2 and len(m.group(2)):
                lambda_arg = m.group(2)
                lambda_ = float(lambda_arg)
            else:
                lambda_ = None

        if lambda_ is None and smoother_name.find("add") != -1:
            raise ValueError(
                f"You must include a non-negative lambda value in smoother name {smoother}"
            )

        if smoother_name == "uniform":
            return UniformLanguageModel()
        elif smoother_name == "add":
            assert lambda_ is not None
            return AddLambdaLanguageModel(lambda_)
        elif smoother_name == "backoff_add":
            assert lambda_ is not None
            return BackoffAddLambdaLanguageModel(lambda_)
        elif smoother_name == "loglinear":
            assert lambda_ is not None
            return LogLinearLanguageModel(lambda_, lexicon)
        elif smoother_name == "improved":
            assert lambda_ is not None
            return ImprovedModel(lambda_, lexicon)
        else:
            raise ValueError(f"Don't recognize smoother name {smoother_name}")

    def file_log_prob(self, corpus: Path) -> float:
        """Compute the log probability of the sequence of tokens in file.
        NOTE: we use natural log for our internal computation.  You will want to
        divide this number by log(2) when reporting log probabilities.
        """
        log_prob = 0.0
        x, y = BOS, BOS
        for z in get_tokens(corpus):
            prob = self.prob(x, y, z)
            log_prob += math.log(prob)
            x, y = y, z  # Shift over by one position.
        return log_prob

    def set_vocab_size(self, *files: Path) -> None:
        if self.vocab is not None:
            log.warning("Warning: vocabulary already set!")

        word_counts: Counter[str] = Counter()  # count of each word

        for file in files:
            for token in get_tokens(file):
                word_counts[token] += 1
                self.show_progress()
        sys.stderr.write("\n")  # done printing progress dots "...."

        vocab: Set[str] = set(w for w in word_counts if word_counts[w] >= OOV_THRESHOLD)
        vocab |= {  # Union equals
            OOV,
            EOS,
        }  # add EOS to vocab (but not BOS, which is never a possible outcome but only a context)

        self.vocab = vocab
        log.info(f"Vocabulary size is {self.vocab_size} types including OOV and EOS")

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    def count(self, x: str, y: str, z: str) -> None:
        """Count the n-grams.  In the perl version, this was an inner function.
        For now, I am just using a data member to store the found tri-
        and bigrams.
        """
        self._count_ngram((x, y, z))
        self._count_ngram((y, z))
        self._count_ngram((z,))
        self._count_ngram(())

    def _count_ngram(self, ngram: Ngram) -> None:
        """Count the n-gram; that is, increment its count in the model."""
        self.tokens[ngram] += 1

    def num_tokens(self, corpus: Path) -> int:
        """Give the number of tokens in the corpus, including EOS."""
        return sum(1 for token in get_tokens(corpus))

    def prob(self, x: str, y: str, z: str) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("Reimplement this in subclasses!")
        raise NotImplementedError(
            f"{class_name} is not implemented yet. (That's your job!)"
        )

    @classmethod
    def load(cls, source: Path) -> "LanguageModel":
        import pickle

        log.info(f"Loading model from {source}")
        with open(source, mode="rb") as f:
            return pickle.load(f)
        log.info(f"Loaded model from {source}")

    def save(self, destination: Path) -> None:
        import pickle

        log.info(f"Saving model to {destination}")
        with open(destination, mode="wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Saved model to {destination}")

    def replace_missing(self, token: str) -> str:
        assert self.vocab is not None
        if token not in self.vocab:
            return OOV
        return token

    def train(self, corpus: Path) -> List[str]:
        """Read the training corpus and collect any information that will be needed
        by the prob function later on.  Tokens are whitespace-delimited.

        Note: In a real system, you wouldn't do this work every time you ran the
        testing program. You'd do it only once and save the trained model to disk
        in some format.
        """
        log.info(f"Training from corpus {corpus}")
        # The real work:
        # accumulate the type and token counts into the global hash tables.

        # If vocab size has not been set, build the vocabulary from training corpus
        if self.vocab is None:
            self.set_vocab_size(corpus)

        # Clear out any previous training.
        self.tokens = Counter()

        # We save the corpus in memory to a list tokens_list.  Notice that we
        # prepended two BOS at the front of the list and appended an EOS at the end.  You
        # will need to add more BOS tokens if you want to use a longer context than
        # trigram.
        x, y = BOS, BOS  # Previous two words.  Initialized as "beginning of sequence"
        # count the BOS context

        self.tokens[(x, y)] = 1
        self.tokens[(y,)] = 1  # The syntax for a 1-element tuple in Python

        tokens_list = [x, y]  # the corpus saved as a list
        for z in get_tokens(corpus):
            z = self.replace_missing(z)
            self.count(x, y, z)
            self.show_progress()
            x, y = y, z  # Shift over by 1 word.
            tokens_list.append(z)
        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.tokens[()]} tokens")
        return tokens_list

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")


class UniformLanguageModel(LanguageModel):
    def prob(self, x: str, y: str, z: str) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(LanguageModel):
    def __init__(self, lambda_: float) -> None:
        super().__init__()

        if lambda_ < 0:
            log.error(f"Lambda value was {lambda_}")
            raise ValueError(
                "You must include a non-negative lambda value in smoother name"
            )
        self.lambda_ = lambda_

    def prob(self, x: str, y: str, z: str) -> float:
        assert self.vocab is not None
        x = self.replace_missing(x)
        y = self.replace_missing(y)
        z = self.replace_missing(z)
        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.
        return (self.tokens[x, y, z] + self.lambda_) / (
                self.tokens[x, y] + self.lambda_ * self.vocab_size
        )


class BackoffAddLambdaLanguageModel(LanguageModel):
    def __init__(self, lambda_: float) -> None:
        super().__init__()

        if lambda_ < 0:
            log.error(f"Lambda value was {lambda_}")
            raise ValueError(
                "You must include a non-negative lambda value in smoother name"
            )
        self.lambda_ = lambda_

    def prob(self, x: str, y: str, z: str) -> float:
        assert self.vocab is not None
        x = self.replace_missing(x)
        y = self.replace_missing(y)
        z = self.replace_missing(z)

        total = self.tokens[()]

        pz = (self.tokens[(z,)] + self.lambda_)/(total + self.lambda_*self.vocab_size)
        pzy = (self.tokens[y,z]+self.lambda_*self.vocab_size*pz)/(self.tokens[(y,)]+self.lambda_*self.vocab_size)

        return (self.tokens[x, y, z] + self.lambda_*self.vocab_size*pzy) / (
            self.tokens[x, y] + self.lambda_ * self.vocab_size
        )


class LogLinearLanguageModel(LanguageModel):
    def __init__(self, c: float, lexicon: Path) -> None:
        super().__init__()

        if c < 0:
            log.error(f"C value was {c}")
            raise ValueError("You must include a non-negative c value in smoother name")
        self.c: float = c
        self.vectors: Dict[str, Vector]
        self.dim: int
        self.vectors, self.dim = self._read_vectors(lexicon)
        self.normalize: float
        self.X: Any = None
        self.Y: Any = None
        self.E: Any = None
        self.denominator: Any = None

    def _read_vectors(self, lexicon: Path) -> Tuple[Dict[str, Vector], int]:
        """Read word vectors from an external file.  The vectors are saved as
        arrays in a dictionary self.vectors.
        """
        with open(lexicon) as f:
            header = f.readline()
            dim = int(header.split()[-1])
            vectors: Dict[str, Vector] = {}
            for line in f:
                word, *arr = line.split()
                vectors[word] = [float(x) for x in arr]

        return vectors, dim

    def u_xyz(self, x:str, y:str, z:str):
        x, y, z = self.replace_missing(x), self.replace_missing(y), self.replace_missing(z)
        x_vector, y_vector, z_vector = np.array(self.vectors[x]), np.array(self.vectors[y]), np.array(self.vectors[z])
        rslt = math.exp(np.dot(np.dot(x_vector, self.X), z_vector[:,np.newaxis]) + np.dot(np.dot(y_vector, self.X), z_vector[:,np.newaxis]))
        return rslt

    def log_u_xyz(self, x:str, y:str, z:str):
        x, y, z = self.replace_missing(x), self.replace_missing(y), self.replace_missing(z)
        x_vector, y_vector, z_vector = np.array(self.vectors[x]), np.array(self.vectors[y]), np.array(self.vectors[z])
        rslt = np.dot(np.dot(x_vector[np.newaxis, :], np.array(self.X)), z_vector[:,np.newaxis]) + np.dot(np.dot(y_vector[np.newaxis, :], np.array(self.Y)), z_vector[:,np.newaxis])
        return rslt

    def Z_xy(self, x:str, y:str):
        x, y = self.replace_missing(x), self.replace_missing(y)
        x_vector, y_vector = np.array(self.vectors[x]), np.array(self.vectors[y])
        rslt = 0
        for w in self.vocab:
            w = self.replace_missing(w)
            rslt += self.u_xyz(x, y, w)
        return rslt

    def log_Z_xy(self, x:str, y:str):
        x, y = self.replace_missing(x), self.replace_missing(y)
        x_vector, y_vector = np.array(self.vectors[x]), np.array(self.vectors[y])
        a = np.zeros(self.vocab_size)
        idx = 0
        for w in self.vocab:
            w = self.replace_missing(w)
            temp = self.log_u_xyz(x, y, w)
            a[idx] = temp
            idx += 1
        return logsumexp(a)

    def replace_missing(self, token: str) -> str:
        # substitute out-of-lexicon words with OOL symbol 
        assert self.vocab is not None
        if token not in self.vocab:
            token = OOV
        if token not in self.vectors:
            token = OOL
        return token

    def prob(self, x: str, y: str, z: str) -> float:
        rslt = math.exp(self.log_u_xyz(x, y, z) - self.log_Z_xy(x, y))
        return rslt

        # x_vector = np.array(self.vectors[x])
        # x_vector_T = x_vector[:,np.newaxis]
        # x_vector = x_vector[np.newaxis,:]
        # y_vector = np.array(self.vectors[y])
        # y_vector_T = y_vector[:, np.newaxis]
        # y_vector = y_vector[np.newaxis, :]
        # z_vector = np.array(self.vectors[z])
        # z_vector_T = z_vector[:, np.newaxis]
        # z_vector = z_vector[np.newaxis, :]
        #
        # numerator = math.exp(x_vector.dot(self.X).dot(z_vector_T) + y_vector.dot(self.Y).dot(z_vector_T))
        # if self.E is None:
        #     self.E = np.zeros((len(self.vectors[z]), len(self.vectors.keys())))
        #     i = 0
        #     for k in self.vocab:
        #         k = self.replace_missing(k)
        #         self.E[:, i] = self.vectors[k]
        #         i+=1

        # if self.denominator is None:
        #     print('z(xy) is renewed\n')
        #     first_term, second_term = x_vector.dot(self.X).dot(self.E), y_vector.dot(self.Y).dot(self.E)
        #     self.denominator = 0
        #     for i in range(len(self.vectors.keys())):
        #         try:
        #             self.denominator += math.exp(first_term[0][i] + second_term[0][i])
        #         except:
        #             print(x_vector.dot(self.X).dot(self.E))
        #             print('\n')
        #             print(self.X)
        #             print(first_term)
        #             print(second_term)
        #             exit()
        # return numerator/self.denominator

    def clear_prob(self) -> None:
        self.E = None
        self.denominator = None

    def train(self, corpus: Path) -> List[str]:
        """Read the training corpus and collect any information that will be needed
        by the prob function later on.  Tokens are whitespace-delimited.

        Note: In a real system, you wouldn't do this work every time you ran the
        testing program. You'd do it only once and save the trained model to disk
        in some format.
        """
        tokens_list = super().train(corpus)  # get list of tokens
        # Train the log-linear model using SGD.

        # Initialize parameters
        self.X = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]
        self.Y = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]

        # Optimization hyperparameters
        gamma0 = 0.01  # initial learning rate, used to compute actual learning rate
        epochs = 10  # number of passes
        t = 0
        self.N = len(tokens_list) - 2  # number of training instances
        # ******** COMMENT *********
        # In log-linear model, you will have to do some additional computation at
        # this point.  You can enumerate over all training trigrams as following.
        #
        #print('start SGD')

        self.E = np.zeros((self.dim, self.vocab_size))
        i = 0
        for k in self.vocab:
            k = self.replace_missing(k)
            self.E[:, i] = self.vectors[k]
            i += 1
        p = 1
        for e in range(epochs):
            F = 0
            for i in range(2, len(tokens_list)):
                # prepare some constants
                x, y, z = tokens_list[i - 2], tokens_list[i - 1], tokens_list[i]
                x, y, z = self.replace_missing(x), self.replace_missing(y), self.replace_missing(z)
                # z_xy = self.Z_xy(x, y)
                log_z = self.log_Z_xy(x, y)
                gamma = gamma0/(1+gamma0*2*self.c/self.N*t)
                x_vector, y_vector, z_vector = np.asarray(self.vectors[x]), np.asarray(self.vectors[y]), np.asarray(self.vectors[z]) # all row vectors

                # compute gradient matrix
                gradient_x, gradient_y= np.zeros([self.dim, self.dim]), np.zeros([self.dim, self.dim])
                gradient_x += np.dot(x_vector[:, np.newaxis], z_vector[np.newaxis,:])
                gradient_y += np.dot(y_vector.reshape(y_vector.shape[0],1), z_vector.reshape(1,z_vector.shape[0]))
                sum = 0
                for z_prime in self.vocab:
                    z_prime = self.replace_missing(z_prime)
                    z_prime_vector = np.array(self.vectors[z_prime])
                    sum += math.exp(self.log_u_xyz(x, y, z_prime) - log_z)
                    gradient_x -= (math.exp(self.log_u_xyz(x, y, z_prime) - log_z))*np.dot(x_vector[:,np.newaxis], z_prime_vector[np.newaxis,:])
                    gradient_y -= (math.exp(self.log_u_xyz(x, y, z_prime) - log_z))*np.dot(y_vector[:,np.newaxis], z_prime_vector[np.newaxis,:])
                gradient_x -= 2 * self.c * np.array(self.X) / self.N
                # if p == 1:
                #     print(gradient_x)
                #     p = 0
                gradient_y -= 2 * self.c * np.array(self.Y) / self.N

                temp_X = np.array(self.X)
                temp_Y = np.array(self.Y)
                temp_X += gamma * np.array(gradient_x)
                temp_Y += gamma * np.array(gradient_y)
                self.X = temp_X.tolist()
                self.Y = temp_Y.tolist()

                #print(x_vector.shape)
                # xXE = np.dot(x_vector.T, self.X).dot(self.E)
                # yYE = np.dot(y_vector.T, self.Y).dot(self.E)
                # u_xyz = 0
                # for k in range(self.vocab_size):
                #     u_xyz += math.exp(xXE[0][k] + yYE[0][k])
                #print(xXE.shape)
                # first_term_x = np.dot(x_vector_T, z_vector)
                # first_term_y = np.dot(y_vector_T, z_vector)
                #
                # second_term_x = np.zeros((len(self.vectors[x]), len(self.vectors[x])))
                # second_term_y = np.zeros((len(self.vectors[x]), len(self.vectors[x])))

                # for j in range(self.dim):
                #     y_j = self.vectors[y][j]
                #     x_j = self.vectors[y][j]
                #     for m in range(self.dim):
                #         z_xy = sum([(math.exp(xXE[0][i] + yYE[0][i])/u_xyz) * x_j * self.E[:, i][m] for i in
                #                     range(self.vocab_size)])
                #         dx = x_j*self.vectors[z][m] - z_xy - (2*self.c/len(tokens_list))*self.X[j][m]
                #         #print(dx)
                #         z_xy = sum([math.exp(xXE[0][i] + yYE[0][i])/u_xyz * y_j * self.E[:, i][m] for i in
                #                     range(self.vocab_size)])
                #         dy = y_j*self.vectors[z][m] - z_xy - 2*self.c/len(tokens_list)*self.Y[j][m]
                #         self.X[j][m] += gamma*dx
                #         self.Y[j][m] += gamma*dy
                        # for z_prime in self.vocab:
                        #     z_prime = self.replace_missing(z_prime)
                        #     second_term_x[j][m] += self.prob(x, y, z_prime) * x_j * self.vectors[z_prime][m]
                        #     second_term_y[j][m] += self.prob(x, y, z_prime) * y_j * self.vectors[z_prime][m]
                        #     self.show_progress()


                # third_term_x = ((2 * self.c)/self.N) * self.X
                # third_term_y = ((2 * self.c)/self.N) * self.Y
                #
                # gradient_x = first_term_x - second_term_x - third_term_x
                # print('\nfirst term: \n')
                # print(first_term_x)
                # print('\n second term: \n')
                # print(second_term_x)
                # print('\n third term: \n')
                # print(third_term_x)
                # print('\ngradient_x: \n')
                # print(gradient_x)
                # print('\n')
                # gradient_y = first_term_y - second_term_y - third_term_y
                # self.X += gamma * gradient_x
                # self.Y += gamma * gradient_y
                # self.clear_prob()
                t += 1
                try:
                    Fi = math.log(self.prob(x, y, z))
                except:
                    Fi = 0
                F += Fi
            F -= self.c * (np.linalg.norm(temp_X) ** 2 + np.linalg.norm(temp_Y) ** 2)
            F = F / self.N
            print(f"epoch {e + 1}: F={F}")
        #
        # Note1: self.c is the regularizer constant C
        # Note2: You can use self.show_progress() to log progress.
        #
        # **************************

        log.info("Start optimizing.")

        #####################
        # TODO: Implement your SGD here
        #####################

        log.info(f"Finished training on {self.tokens[()]} tokens")
        return tokens_list  # Not really needed, except to obey typing.


class ImprovedModel(LanguageModel):
    def __init__(self, c: float, lexicon: Path) -> None:
        super().__init__()

        self.beta = None
        if c < 0:
            log.error(f"C value was {c}")
            raise ValueError("You must include a non-negative c value in smoother name")
        self.c: float = c
        self.vectors: Dict[str, Vector]
        self.dim: int
        self.vectors, self.dim = self._read_vectors(lexicon)
        self.normalize: float
        self.X: Any = None
        self.Y: Any = None
        self.E: Any = None
        self.denominator: Any = None

    def _read_vectors(self, lexicon: Path) -> Tuple[Dict[str, Vector], int]:
        """Read word vectors from an external file.  The vectors are saved as
        arrays in a dictionary self.vectors.
        """
        with open(lexicon) as f:
            header = f.readline()
            dim = int(header.split()[-1])
            vectors: Dict[str, Vector] = {}
            for line in f:
                word, *arr = line.split()
                vectors[word] = [float(x) for x in arr]

        return vectors, dim

    def u_xyz(self, x:str, y:str, z:str):
        x, y, z = self.replace_missing(x), self.replace_missing(y), self.replace_missing(z)
        x_vector, y_vector, z_vector = np.array(self.vectors[x]), np.array(self.vectors[y]), np.array(self.vectors[z])
        rslt = math.exp(np.dot(np.dot(x_vector, self.X), z_vector[:,np.newaxis]) + np.dot(np.dot(y_vector, self.X), z_vector[:,np.newaxis]))
        return rslt

    def log_u_xyz(self, x:str, y:str, z:str):
        x, y, z = self.replace_missing(x), self.replace_missing(y), self.replace_missing(z)
        x_vector, y_vector, z_vector = np.array(self.vectors[x]), np.array(self.vectors[y]), np.array(self.vectors[z])
        rslt = np.dot(np.dot(x_vector[np.newaxis, :], np.array(self.X)), z_vector[:,np.newaxis]) + np.dot(np.dot(y_vector[np.newaxis, :], np.array(self.Y)), z_vector[:,np.newaxis]) + self.beta * self.unigram(x, y, z)
        return rslt

    def Z_xy(self, x:str, y:str):
        x, y = self.replace_missing(x), self.replace_missing(y)
        x_vector, y_vector = np.array(self.vectors[x]), np.array(self.vectors[y])
        rslt = 0
        for w in self.vocab:
            w = self.replace_missing(w)
            rslt += self.u_xyz(x, y, w)
        return rslt

    def log_Z_xy(self, x:str, y:str):
        x, y = self.replace_missing(x), self.replace_missing(y)
        x_vector, y_vector = np.array(self.vectors[x]), np.array(self.vectors[y])
        a = np.zeros(self.vocab_size)
        idx = 0
        for w in self.vocab:
            w = self.replace_missing(w)
            temp = self.log_u_xyz(x, y, w)
            a[idx] = temp
            idx += 1
        return logsumexp(a)

    def replace_missing(self, token: str) -> str:
        # substitute out-of-lexicon words with OOL symbol
        assert self.vocab is not None
        if token not in self.vocab:
            token = OOV
        if token not in self.vectors:
            token = OOL
        return token

    def unigram(self, x: str, y: str, z: str):
        return math.log(self.tokens[z,] + 1)

    def prob(self, x: str, y: str, z: str) -> float:
        rslt = math.exp(self.log_u_xyz(x, y, z) - self.log_Z_xy(x, y))
        return rslt

    def clear_prob(self) -> None:
        self.E = None
        self.denominator = None

    def train(self, corpus: Path) -> List[str]:
        """Read the training corpus and collect any information that will be needed
        by the prob function later on.  Tokens are whitespace-delimited.

        Note: In a real system, you wouldn't do this work every time you ran the
        testing program. You'd do it only once and save the trained model to disk
        in some format.
        """
        tokens_list = super().train(corpus)  # get list of tokens
        # Train the log-linear model using SGD.

        # Initialize parameters
        self.X = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]
        self.Y = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]
        self.beta = 0

        # Optimization hyperparameters
        gamma0 = 0.01  # initial learning rate, used to compute actual learning rate
        epochs = 10  # number of passes
        t = 0
        self.N = len(tokens_list) - 2  # number of training instances
        # ******** COMMENT *********
        # In log-linear model, you will have to do some additional computation at
        # this point.  You can enumerate over all training trigrams as following.
        #
        #print('start SGD')

        self.E = np.zeros((self.dim, self.vocab_size))
        i = 0
        for k in self.vocab:
            k = self.replace_missing(k)
            self.E[:, i] = self.vectors[k]
            i += 1
        p = 1
        for e in range(epochs):
            F = 0
            for i in range(2, len(tokens_list)):
                # prepare some constants
                x, y, z = tokens_list[i - 2], tokens_list[i - 1], tokens_list[i]
                x, y, z = self.replace_missing(x), self.replace_missing(y), self.replace_missing(z)
                # z_xy = self.Z_xy(x, y)
                log_z = self.log_Z_xy(x, y)
                gamma = gamma0/(1+gamma0*2*self.c/self.N*t)
                x_vector, y_vector, z_vector = np.asarray(self.vectors[x]), np.asarray(self.vectors[y]), np.asarray(self.vectors[z]) # all row vectors

                # compute gradient matrix
                gradient_x, gradient_y= np.zeros([self.dim, self.dim]), np.zeros([self.dim, self.dim])
                gradient__beta = 0
                gradient_x += np.dot(x_vector[:, np.newaxis], z_vector[np.newaxis,:])
                gradient_y += np.dot(y_vector.reshape(y_vector.shape[0],1), z_vector.reshape(1,z_vector.shape[0]))
                gradient__beta += self.unigram(x, y, z)
                sum = 0
                for z_prime in self.vocab:
                    z_prime = self.replace_missing(z_prime)
                    z_prime_vector = np.array(self.vectors[z_prime])
                    sum += math.exp(self.log_u_xyz(x, y, z_prime) - log_z)
                    gradient_x -= (self.prob(x, y, z_prime))*np.dot(x_vector[:,np.newaxis], z_prime_vector[np.newaxis,:])
                    gradient_y -= (self.prob(x, y, z_prime))*np.dot(y_vector[:,np.newaxis], z_prime_vector[np.newaxis,:])
                    gradient__beta -= self.prob(x, y, z_prime) * self.unigram(x, y, z_prime)
                gradient_x -= 2 * self.c * np.array(self.X) / self.N
                # if p == 1:
                #     print(gradient_x)
                #     p = 0
                gradient_y -= 2 * self.c * np.array(self.Y) / self.N
                gradient__beta -= 2 * self.c * np.array(self.beta) / self.N

                temp_X = np.array(self.X)
                temp_Y = np.array(self.Y)
                temp_X += gamma * np.array(gradient_x)
                temp_Y += gamma * np.array(gradient_y)
                self.X = temp_X.tolist()
                self.Y = temp_Y.tolist()
                self.beta += gamma * gradient__beta
                #print(gradient__beta)
                #print(x_vector.shape)
                # xXE = np.dot(x_vector.T, self.X).dot(self.E)
                # yYE = np.dot(y_vector.T, self.Y).dot(self.E)
                # u_xyz = 0
                # for k in range(self.vocab_size):
                #     u_xyz += math.exp(xXE[0][k] + yYE[0][k])
                #print(xXE.shape)
                # first_term_x = np.dot(x_vector_T, z_vector)
                # first_term_y = np.dot(y_vector_T, z_vector)
                #
                # second_term_x = np.zeros((len(self.vectors[x]), len(self.vectors[x])))
                # second_term_y = np.zeros((len(self.vectors[x]), len(self.vectors[x])))

                # for j in range(self.dim):
                #     y_j = self.vectors[y][j]
                #     x_j = self.vectors[y][j]
                #     for m in range(self.dim):
                #         z_xy = sum([(math.exp(xXE[0][i] + yYE[0][i])/u_xyz) * x_j * self.E[:, i][m] for i in
                #                     range(self.vocab_size)])
                #         dx = x_j*self.vectors[z][m] - z_xy - (2*self.c/len(tokens_list))*self.X[j][m]
                #         #print(dx)
                #         z_xy = sum([math.exp(xXE[0][i] + yYE[0][i])/u_xyz * y_j * self.E[:, i][m] for i in
                #                     range(self.vocab_size)])
                #         dy = y_j*self.vectors[z][m] - z_xy - 2*self.c/len(tokens_list)*self.Y[j][m]
                #         self.X[j][m] += gamma*dx
                #         self.Y[j][m] += gamma*dy
                        # for z_prime in self.vocab:
                        #     z_prime = self.replace_missing(z_prime)
                        #     second_term_x[j][m] += self.prob(x, y, z_prime) * x_j * self.vectors[z_prime][m]
                        #     second_term_y[j][m] += self.prob(x, y, z_prime) * y_j * self.vectors[z_prime][m]
                        #     self.show_progress()


                # third_term_x = ((2 * self.c)/self.N) * self.X
                # third_term_y = ((2 * self.c)/self.N) * self.Y
                #
                # gradient_x = first_term_x - second_term_x - third_term_x
                # print('\nfirst term: \n')
                # print(first_term_x)
                # print('\n second term: \n')
                # print(second_term_x)
                # print('\n third term: \n')
                # print(third_term_x)
                # print('\ngradient_x: \n')
                # print(gradient_x)
                # print('\n')
                # gradient_y = first_term_y - second_term_y - third_term_y
                # self.X += gamma * gradient_x
                # self.Y += gamma * gradient_y
                # self.clear_prob()
                t += 1
                try:
                    Fi = math.log(self.prob(x, y, z))
                except:
                    Fi = 0
                F += Fi
            F -= self.c * (np.linalg.norm(temp_X) ** 2 + np.linalg.norm(temp_Y) ** 2)
            F = F / self.N
            print(f"epoch {e + 1}: F={F}")
        #
        # Note1: self.c is the regularizer constant C
        # Note2: You can use self.show_progress() to log progress.
        #
        # **************************

        log.info("Start optimizing.")

        #####################
        # TODO: Implement your SGD here
        #####################

        log.info(f"Finished training on {self.tokens[()]} tokens")
        return tokens_list  # Not really needed, except to obey typing.