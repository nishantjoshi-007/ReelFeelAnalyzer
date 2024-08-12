# :star2: ReelFeel Analyzer: Sentiment Analysis on IMDB Movie Reviews

<div align='center'>

<p>Welcome to the 'ReelFeel Analyzer' GitHub repository! This project involves performing sentiment analysis on the IMDB Movie Reviews Dataset. The primary objective is to classify movie reviews as positive or negative based on the content of the textual data. The project covers several key areas, including text preprocessing, classification using machine learning algorithms, and model evaluation.</p>
<h4> <a href="https://github.com/nishantjoshi-007/ReelFeelAnalyzer/blob/main/static/Report.pdf"> Documentation </a> <span> · </span> <a href="https://github.com/nishantjoshi-007/ReelFeelAnalyzer/issues"> Report Bug </a>

</div>



# :notebook_with_decorative_cover: Table of Contents
- [Getting Started](#toolbox-getting-started)
- [Prerequisites](#bangbang-prerequisites)
- [Project Implementation](#gear-project-implementation)
- [Contributing](#wave-contributing)
- [License](#warning-license)




## :toolbox: Getting Started
### Project Structure
The project is structured as follows:
1. **Project Description**: Overview of the project objectives and goals.
2. **Data Loading and Preprocessing**: Loading the dataset, cleaning the text data by removing HTML tags, converting to lowercase, and removing punctuation.
3. **Text Vectorization**: Converting text data to numerical features using TF-IDF.
4. **Model Training**: Splitting the dataset into training and testing sets, and training a Naive Bayes classifier.
5. **Hyperparameter Tuning**: Using GridSearchCV for optimizing hyperparameters.
6. **Model Evaluation**: Evaluating the model's performance using accuracy, precision, recall, F1-score, confusion matrix, ROC curve, Precision-Recall curve, and distribution of predicted probabilities.
7. **Visualization**: Visualizing the results using word clouds and distribution plots.
8. **Error Analysis**: Analyzing misclassified examples to understand model weaknesses.
9. **Conclusion**: Summarizing findings and suggesting potential future improvements.


## :bangbang: Prerequisites
### Dataset
The dataset used for this project is the [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), which contains 50,000 movie reviews, with 25,000 reviews for training and 25,000 reviews for testing. The dataset is balanced, containing an equal number of positive and negative reviews.
### Dependencies
To run this project, you need the following dependencies:
- pandas
- numpy
- re
- BeautifulSoup
- matplotlib
- seaborn
- scikit-learn
- wordcloud

You can install these dependencies using `pip`:
```bash
pip install -r requirements.txt
```

## :gear: Project Implementation
### Data Loading and Preprocessing
- Load the dataset.
- Clean the text data by removing HTML tags, converting to lowercase, and removing punctuation.
### Text Vectorization
- Convert text data to numerical features using TF-IDF.
### Model Training and Hyperparameter Tuning
- Split the dataset into training and testing sets.
- Train a Naive Bayes classifier.
- Optimize hyperparameters using GridSearchCV.
### Model Evaluation
- Evaluate the model's performance using accuracy, precision, recall, F1-score, and confusion matrix.
- Generate and plot Precision-Recall Curve, ROC Curve, and the distribution of predicted probabilities.
### Visualization
- Generate word clouds for positive and negative reviews.
- Plot the distribution of review lengths.
### Error Analysis
- Analyze misclassified examples to understand model weaknesses.
### Conclusion
- Summarize findings and suggest potential future improvements.


## :wave: Contributing
<img src="https://contrib.rocks/image?repo=Louis3797/awesome-readme-template" /> Contributions to the ReelFeel Analyzer are always welcome! Whether it's reporting bugs, suggesting new features, or improving the code, your input is valuable. Please feel free to fork this repository, make your changes, and submit a pull request.


## :warning: License
Distributed under the MIT License. See <a href="https://github.com/nishantjoshi-007/ReelFeelAnalyzer/blob/main/LICENSE">LICENSE.txt</a> for more information.