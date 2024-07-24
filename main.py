import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pickle


model=load_model('next_word_lstm.h5')

with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)
    
#function to predict
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    token_list=pad_sequences([token_list],maxlen=max_sequence_len,padding='pre')
    predicted=model.predict(token_list)
    predict_word_index=np.argmax(predicted,axis=1)
    for word, index in tokenizer.word_index.items():
        if index==predict_word_index:
            return word
        
#Streamlit app
st.title('Next Word Prediction Algo by LSTM')
input_text=st.text_area('enter text')
if st.button('predict'):
    output_predicted=predict_next_word(model,tokenizer,text=input_text,max_sequence_len=13)
    st.write('Next word : ',output_predicted)
 
    
    

