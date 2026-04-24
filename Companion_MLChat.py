import joblib 


bow_vectorizer_lem = joblib.load('bow_vectorizer_lem.pkl')
xgb_bow_lem = joblib.load('depression_detection_model.pkl')

print("AI system: Hello! I'm here to help.let us start our session by asking, how are you today?")
counter=0
new_text=""
new_texts=[]
while True:
    user_input = input("You: ")
    counter+=1
    # Check for exit command
    if user_input.lower() in ['q', 'quit', 'exit']:
        print("AI system: I'm glad we chatted! Take care, and goodbye!")
        break

    if user_input.lower() not in ['q', 'quit', 'exit'] and counter==1:
        new_text=user_input
        print("AI system: Please give me a few more details into how you feel?")

    if user_input.lower() not in ['q', 'quit', 'exit'] and counter==2:
        new_texts.append(new_text+" "+user_input)
        X_new_bow = bow_vectorizer_lem.transform(new_texts)
        y_pred_new = xgb_bow_lem.predict(X_new_bow)
        if y_pred_new == 1 :
           print("AI system: It must be tough for you, rest assured we are here to assist, i will refere you to our Virtual Counselor in a second")
           break
        else:
            print("AI system: It seams you are not showing any signs of depression, you can talk to a professional if you feel differrent, or get some other help")
            break

