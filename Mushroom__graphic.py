
#program will use sklearn module for machine training. we import modules 

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score 

from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Load and preprocess data
# first we load df files 
df = pd.read_csv("mushroom.csv")
#to correct csv file 
df=df.applymap(lambda x: x.strip() if isinstance(x, str) else x) 

#instead of manual labeling we use LabelEncoder
color_encoder = LabelEncoder()
df['Color'] = color_encoder.fit_transform(df['Color'])

appearance_encoder = LabelEncoder()
df['Appearance'] = appearance_encoder.fit_transform(df['Appearance'])

smell_encoder = LabelEncoder()
df['Smell'] = smell_encoder.fit_transform(df['Smell'])

poisoned_encoder = LabelEncoder()
df['Poisoned'] = poisoned_encoder.fit_transform(df['Poisoned'])
#Split data as X,y : it means features and output 

X = df[['Color', 'Appearance', 'Smell']] #features 
y = df['Poisoned'] #output yes or no 

#train and test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

#create confusion matrix 

cm = confusion_matrix(y_test, y_pred)

#functions below used for tkinter 

def plot_confusion_matrix():
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=poisoned_encoder.classes_).plot(ax=ax)
    return fig

def plot_tree_graph():
    fig, ax = plt.subplots(figsize=(15, 10)) #10,7
    plot_tree(dtree, feature_names=['Color', 'Appearance', 'Smell'], filled=True, ax=ax)
    ax.set_title(f"Decision Tree\nAccuracy: {accuracy:.2f}")
    return fig


def predict_mushroom():
    color = color_encoder.transform([color_var.get()])
    appearance = appearance_encoder.transform([appearance_var.get()])
    smell = smell_encoder.transform([smell_var.get()])
    prediction = dtree.predict([[color[0], appearance[0], smell[0]]])

    # Update this line to customize the output text based on the prediction
    prediction_text = "Yes, it is poisoned" if poisoned_encoder.inverse_transform(prediction)[0] == 'Yes' else "No, it is not poisoned"
    result.set(prediction_text)


def show_tree_separately():
    # Create new window for seperate decisiontree
    new_win = Toplevel(root)
    new_win.title("Decision Tree")
    
    # Plot the decision tree in the new window
    canvas = FigureCanvasTkAgg(plot_tree_graph(), master=new_win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)



#GRAPHIC INTERFACE 

#TK INTER
root = Tk()
root.title("Mushroom Predictor")
root.geometry("1400x1250")

# Set a general background color
root.configure(bg='lightgrey')

main_frame = Frame(root, bg='lightgrey')
main_frame.pack(padx=10, pady=10, fill='both', expand=True)

left_frame = Frame(main_frame, bg='lightgrey')
left_frame.pack(side='left', padx=10, pady=10, fill='both', expand=True)

right_frame = Frame(main_frame, bg='lightgrey')
right_frame.pack(side='right', padx=10, pady=10, fill='y')

# Label for Raw Data
Label(left_frame, text="Past Data", bg='lightgrey', font=('Arial', 16)).pack(pady=10)
left_frame.place(x=0,y=0,width=1000,relheight=1)

# Display data in Treeview with height adjusted
treeview = ttk.Treeview(left_frame, height=10, columns=('Color', 'Appearance', 'Smell', 'Poisoned'), show='headings')
treeview.heading('Color', text='Color')
treeview.heading('Appearance', text='Appearance')
treeview.heading('Smell', text='Smell')
treeview.heading('Poisoned', text='Poisoned')

treeview.column('Color', width=50)
treeview.column('Appearance', width=50)
treeview.column('Smell', width=50)
treeview.column('Poisoned', width=50)
treeview.pack(fill='x', pady=10)

for index, row in df.iterrows():
    color = 'red' if poisoned_encoder.inverse_transform([row['Poisoned']])[0] == 'Yes' else 'green'
    treeview.insert('', 'end', values=(color_encoder.inverse_transform([row['Color']])[0],
                                       appearance_encoder.inverse_transform([row['Appearance']])[0],
                                       smell_encoder.inverse_transform([row['Smell']])[0],
                                       poisoned_encoder.inverse_transform([row['Poisoned']])[0]),
                                       tags=('poisoned' if color == 'red' else 'not_poisoned'))

treeview.tag_configure('poisoned', foreground='red')
treeview.tag_configure('not_poisoned', foreground='green')



# Label for Confusion Matrix Explanation
explanation = """The confusion matrix visualizes
the performance of the algorithm.
True Positives (TP) are at the top-left,
True Negatives (TN) are at the bottom-right.
False Positives (FP) are at the top-right,
and False Negatives (FN) are at the bottom-left."""


conf_exp_label=Label(root,text=explanation, bg='lightgrey', font=('Arial', 8))
conf_exp_label.place(x=20,y=450)

# Plot graphs with size adjustments
canvas = FigureCanvasTkAgg(plot_confusion_matrix(), master=left_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill='x', pady=10, padx=5)

canvas2 = FigureCanvasTkAgg(plot_tree_graph(), master=left_frame)
canvas2.draw()
canvas2.get_tk_widget().pack(fill='x', pady=10, padx=5)


# Change the dropdown background
style = ttk.Style()
style.map('TCombobox', selectbackground=[('readonly', 'cyan')])

# Dropdown for prediction in right frame
color_var = StringVar()
appearance_var = StringVar()
smell_var = StringVar()
result = StringVar()

Label(right_frame, text="Predict Mushroom Type", bg='lightgrey', font=('Arial', 16)).pack(pady=10)
Label(right_frame, text="Choose Color:", bg='lightgrey')  #pack(anchor='w')
right_frame.place(x=1000,y=10)
color_dropdown = ttk.Combobox(right_frame, textvariable=color_var, values=list(color_encoder.classes_))
color_dropdown.pack(fill='x', padx=5, pady=2)

Label(right_frame, text="Choose Appearance:", bg='lightgrey').pack(anchor='w')
appearance_dropdown = ttk.Combobox(right_frame, textvariable=appearance_var, values=list(appearance_encoder.classes_))
appearance_dropdown.pack(fill='x', padx=5, pady=2)

Label(right_frame, text="Choose Smell:", bg='lightgrey').pack(anchor='w')
smell_dropdown = ttk.Combobox(right_frame, textvariable=smell_var, values=list(smell_encoder.classes_))
smell_dropdown.pack(fill='x', padx=5, pady=2)

predict_button = Button(right_frame, text="Predict", command=predict_mushroom, bg='cyan')
predict_button.pack(pady=10)

result_label = Label(right_frame, textvariable=result, font=("Arial", 12), bg='lightgrey')
result_label.pack(pady=10)

#show tree seperatetly button
tree_button = Button(left_frame, text="View Tree Separately", command=show_tree_separately, bg='cyan')
tree_button.place(x=10,y=10)


root.mainloop()

