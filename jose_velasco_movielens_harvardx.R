##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("stringr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(stringr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#create additional column with the release year of the movie
#in order to expand the analysis 
edx <- edx %>% 
  mutate(year = as.numeric(str_sub(title,start= -5,end= -2)))
validation <- validation %>%
  mutate(year = as.numeric(str_sub(title,start= -5,end= -2)))

#we validate the years are generated correctly
#no missing data
summary(edx$year)
summary(validation$year)

#####exploring the dataset#######
dim(as.matrix(edx)) #we learn the dimension of the dataset
names(edx) #as well as the names of the variables

#to get an idea of how many users and movies are in the dataset
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#we also look at the distribution of movies and users
p1 <- edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(fill = "#FC4E08",color = "grey", binwidth = 0.25) + 
  scale_x_log10() + 
  geom_density(aes(y=..count..*0.25), alpha = 0.5, color="blue", 
               linetype ="dashed", size =0.5) +
  labs(x = "Movies", y = "Number of Ratings") + 
  ggtitle("Movie Ratings by Movie") 
p2 <- edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(fill = "#FC4E08",color = "grey", binwidth = 0.25) + 
  scale_x_log10() + 
  geom_density(aes(y=..count..*0.25), alpha = 0.5, color="blue", 
               linetype ="dashed", size =0.5) +
  labs(x = "Users", y = "Number of Ratings") + 
  ggtitle("Movie Ratings by User") 
gridExtra::grid.arrange(p1, p2, ncol = 2)

# make a plot by movie release date and number of ratings
edx %>% count(year) %>% 
  ggplot(aes(y =n/1000, x = year)) + 
  geom_line(color = "#FC4E08") + 
  labs(x = "Year", y = "Ratings (thousands)") + 
  ggtitle("Year of release and number of ratings") 
######DEFINING RMSE#######
#we define a function to calculate our RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#####BASELINE MODEL######
#we just take the mean
mu <- mean(edx$rating)
mu

#and we store the RMSE which will e our benchmark
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse
rmse_results <- tibble(method = "Baseline model", RMSE = naive_rmse)

####MODEL_1#######
#we take the mean for each movie 
 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
#we see how it is distributed accross the mean
qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))

#now we predict with this new model
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
RMSE(predicted_ratings, validation$rating)

#and store the results
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",  
                                 RMSE = model_1_rmse))

######MODEL_2######
#Now we model the user effects
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#and predict with this new model
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
RMSE(predicted_ratings, validation$rating)

#finally we store the results

model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effects Model",  
                                 RMSE = model_2_rmse))

######MODEL_3######
#Now we model the year effect
year_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u))

#and predict with this new model
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  mutate(pred = mu + b_i + b_u + b_y) %>%
  pull(pred)
RMSE(predicted_ratings, validation$rating)

#finally we store the results

model_3_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User + Year Effects Model",  
                                 RMSE = model_3_rmse))


#####REGULARIZED MODEL#######
#a penalization function is programmed in order to give more weight to movies 
#with many ratings and less sd and give less weight to movies witn few ratings

#cross validation process to generate the best lambda

set.seed(1998)
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, test_set)
edx <- rbind(train_set, removed)

rm(test_index, temp, removed)


#we can find the best lambda for movies and users 
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
   
  mu_train <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_train)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_train)/(n()+l))
  
  b_y <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - b_i - b_u - mu_train)/(n()+l))
  
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "year") %>%
    mutate(pred = mu_train + b_i + b_u + b_y) %>%
    pull(pred)
  
  return(RMSE(as.matrix(test_set$rating), as.matrix(predicted_ratings)))
})
qplot(lambdas, rmses)  


#and the full model for the optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda

#now its time to predict ratings and compute rmse

model_4_rmse <- min(rmses)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User + Year Effect Model",  
                                 RMSE = model_4_rmse))
rmse_results 



