import streamlit as st

# Text preprocessing packages
import nltk # Text libarary
# nltk.download('stopwords')
import string # Removing special characters {#, @, ...}
import re # Regex Package

# Corpora is a group presenting multiple collections of text documents. A single collection is called corpus.
# Stopwords
from nltk.corpus import stopwords

# Stemmer & Lemmatizer
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Text Embedding
from sklearn.feature_extraction.text import TfidfVectorizer

#To Load The Model
import pickle



def raw_test(review, model, vectorizer):
    # Clean Review
    review_c = cleaning_text(review)
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"

def cleaning_text(Text):
    stop_words = list(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    lemmatizer= WordNetLemmatizer()
    #Removing every not from stop words
    stop_words.remove('not')
    for i in stop_words:
        if "n't" in i:
            stop_words.remove(i)

    #Removing Stop words from the Text        
    Text=[i for i in str(Text).split() if i not in stop_words]

    #Removing special characters
    Text=[re.sub('[^A-Za-z0-9]+', '', str(i)) for i in Text]

    #lemmatizing each word
    Text=[lemmatizer.lemmatize(y) for y in Text]

    #stemming each word
    Text=[stemmer.stem(y) for y in Text]
    
    str1 = " " 
    Cleaned_Text=str1.join(Text)
    #Remove numbers
    Cleaned_Text=''.join([i for i in Cleaned_Text if not i.isdigit()])
    # return string  
    return (Cleaned_Text)

    



def main():

    #NLp app with sreamlit

    #here we will load the model we created before
    model_name = 'rf_model.pk'
    vectorizer_name = 'tfidf_vectorizer.pk'
    loaded_model = pickle.load(open(model_name, 'rb'))
    loaded_vect = pickle.load(open(vectorizer_name, 'rb'))
     
    #we will set title  
    st.title("Amazon Food Review")

    #we will make a text area and will let the user to input the text
    #and we will display the output if it positive or negative
    message=st.text_area("Enter Your Review","Type Here ..")
    if st.button("Analyze"):
        clean_result=raw_test(message, loaded_model, loaded_vect)
        st.success(clean_result)
        



if __name__=='__main__':

    main()    




