import time
import tkinter as tk
from tkinter import ttk
from torch import nn
import torch.nn as nn
from sentence_transformers import models
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import kmeans_test

class ShroomForwardFormer(nn.Module):
    def __init__(self,max_seq_length):
        super(ShroomForwardFormer, self).__init__()
        self.word_embedding_model = models.Transformer("distilbert-base-uncased", max_seq_length=max_seq_length)
        self.pooling_model = models.Pooling(self.word_embedding_model.get_word_embedding_dimension())
        self.dense_model = models.Dense(
            in_features=self.pooling_model.get_sentence_embedding_dimension(),
            out_features=256,
            activation_function=nn.Tanh(),
        )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward_once(self, x):
        embedding = self.word_embedding_model(x)
        embedding = self.pooling_model(embedding)
        embedding = self.dense_model(embedding)
        return embedding

    def forward(self, sentence1, sentence2):
        output1 = self.forward_once(sentence1)["sentence_embedding"]
        output2 = self.forward_once(sentence2)["sentence_embedding"]
        similarity_score = self.cos(output1, output2)
        return similarity_score

state_dict = torch.load("best_model_v2_lr1.pth")
model=ShroomForwardFormer(256)
model.load_state_dict(state_dict)

def encode_pair(pair,padding=True,truncation=True,max_length=256):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(
    pair[0], padding=padding,truncation=truncation,max_length=max_length, return_tensors="pt"
    ), tokenizer(
        pair[1],padding=padding,truncation=truncation,max_length=max_length, return_tensors="pt"
    )


class CardEvaluationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Model Card")
        self.label_font = ("Helvetica", 20, "bold")
        self.label_font2 = ("Helvetica", 15, "bold")
        self.label_font3 = ("Helvetica",20, "bold")


        self.configure(background="#EBF3E8")
        options = ["DM", "MT", "PG"]
           # Add radio buttons
        # self.radio_frame = tk.Frame(self, background="#EBF3E8")
        # self.radio_frame.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        # for option in options:
            # rb.pack(side="left", padx=(0, 10))
        self.selected_radio = tk.StringVar()
        self.selected_radio.set("Siamese") 
        self.rb = tk.Radiobutton(self, text="Siamese", variable=self.selected_radio, value="Siamese", background="#EBF3E8",command=self.radio_button_pressed, font=self.label_font3, fg="#86A789")
        self.rb.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.rb = tk.Radiobutton(self, text="Clustering", variable=self.selected_radio, value="Clustering", background="#EBF3E8",command=self.radio_button_pressed, font=self.label_font3, fg="#86A789")
        self.rb.grid(row=0, column=0, padx=10, pady=5, sticky="e")
        



        self.selected_option = tk.StringVar(self)
        self.selected_option.set(options[0]) 
        self.hypothesis_label = tk.Label(self, text="Task",font=self.label_font,foreground="#86A789",background="#EBF3E8")
        self.hypothesis_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.dropdown = tk.OptionMenu(self, self.selected_option, *options, command=self.on_select)
        self.dropdown.grid(row=2, column=0, padx=10, pady=5, sticky="ew") 
        self.task=None
        
        self.dropdown.config(font=self.label_font3,bg="#EBF3E8",fg="#86A789")
        self.dropdown.config(activebackground="black",activeforeground="#D2E3C8") 


        self.hypothesis_label = tk.Label(self, text="Source",font=self.label_font,foreground="#86A789",background="#EBF3E8")
        self.hypothesis_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.source_entry = tk.Text(self, width=50, height=5,insertbackground="#86A789",background="#D2E3C8",foreground="#344955",font=self.label_font2)
        self.source_entry.grid(row=4, column=0, padx=10, pady=5)

        self.hypothesis_label = tk.Label(self, text="Hypothesis",font=self.label_font,foreground="#86A789",background="#EBF3E8")
        self.hypothesis_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.hypothesis_entry = tk.Text(self, width=50, height=2,insertbackground="#86A789",background="#D2E3C8",foreground="#344955",font=self.label_font2)
        self.hypothesis_entry.grid(row=6, column=0, padx=10, pady=5)

        self.hypothesis_label = tk.Label(self, text="Target",font=self.label_font,foreground="#86A789",background="#EBF3E8")
        self.hypothesis_label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.target_entry = tk.Text(self, width=50, height=2,insertbackground="#86A789",background="#D2E3C8",foreground="#344955",font=self.label_font2)
        self.target_entry.grid(row=8, column=0, padx=10, pady=5)

        self.button = tk.Button(self, text="Evaluate",font=self.label_font3, command=self.on_button_click,background="#EBF3E8", foreground="#86A789")
        self.button.grid(row=9, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.hallu_bar=None
        self.semantic_label=None
        self.hallucination_label=None
        self.not_hallucination=None
        self.columnconfigure(0, weight=1)   
        self.center_window()

    
    def on_select(self,value):
        pass
      
        # print("Selected option:", self.selected_option.get())

    def radio_button_pressed(self):
        pass
  
    
        # print(kmeans_test.shroom_predict("I do not need to remind you of the decisive role that Parliament played in the mad cow crisis.", "","Parliament played an important role in the mad cow crisis.", "PG"))

    def hide_progress_bar(self):
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.after(1000, self.create_progess_bar)  

    def hide_progress_bar2(self):
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.after(1000, self.create_progess_bar2)  

    def show_selected(self):
     print("Selected option:", self.selected_option.get())
  
# print(
#     shroom_predict("I do not need to remind you of the decisive role that Parliament played in the mad cow crisis.", "",
#                    "Parliament played an important role in the mad cow crisis.", "PG"))
# print(shroom_predict("\u0174acali kuwonawona kakuliro ka ngozi kweniso na umo caro capasi ciikhwaskikirenge.",
#                      "They are still trying to determine just how large the crash was and how the Earth will be affected.",
#                      "We are still seeing the increasing danger and the impact this planet will have.", "MT"))


    def create_progess_bar(self):

        self.hallucination_label = tk.Label(self, text="Hallucination",font=self.label_font,foreground="#86A789",background="#EBF3E8")
        self.hallucination_label.grid(row=10, column=0, padx=10, pady=5, sticky="w")

        self.not_hallucination = tk.Label(self, text="Not-Hallucination",font=self.label_font,foreground="#86A789",background="#EBF3E8")
        self.not_hallucination.grid(row=10, column=0, padx=10, pady=5, sticky="e")
        style = ttk.Style() 
        style.theme_use('clam') 
      
        if(self.similarity<0.7):
            style.configure("Custom.Horizontal.TProgressbar", troughcolor='#86A789', background='#FA7070') # Change trough and bar color
        else:
             style.configure("Custom.Horizontal.TProgressbar", troughcolor='#86A789', background='#C6EBC5') # Change trough and bar color


        self.hallu_bar = ttk.Progressbar(self, orient='horizontal', length=200,style="Custom.Horizontal.TProgressbar")
        self.hallu_bar.grid(row=12, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.hallu_bar.step(self.similarity*100)

        text="Semantic similarity: "+ str(self.similarity)
        self.semantic_label = tk.Label(self, text=text,font=self.label_font2,foreground="#86A789",background="#EBF3E8")
        self.semantic_label.grid(row=13, column=0, padx=10, pady=5, sticky="w")

    def create_progess_bar2(self):
        text="Ensemble clustering prediction: "+ str(self.similarity2)
        self.semantic_label = tk.Label(self, text=text,font=self.label_font2,foreground="#86A789",background="#EBF3E8")
        self.semantic_label.grid(row=13, column=0, padx=10, pady=5, sticky="w")


    def on_button_click(self):
  
        if(self.hallu_bar!=None):
            self.hallu_bar.grid_remove()
        if(self.semantic_label!=None):
            self.semantic_label.grid_remove()
        if(self.hallucination_label!=None):
            self.hallucination_label.grid_remove()
        if(self.not_hallucination!=None):
            self.not_hallucination.grid_remove()

        style = ttk.Style() 
        style.theme_use('clam')  
        style.configure("Custom.Horizontal.TProgressbar", troughcolor='#86A789', background='#D2E3C8') # Change trough and bar color
        self.progress_bar = ttk.Progressbar(self, orient='horizontal', length=200, mode='indeterminate',style="Custom.Horizontal.TProgressbar")
        
        self.progress_bar.start()

        self.progress_bar.grid(row=10, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        hyp = self.hypothesis_entry.get("1.0",'end-1c')
        tgt = self.target_entry.get("1.0",'end-1c')
        if self.selected_radio.get()=="Clustering":
            src=self.source_entry.get("1.0",'end-1c')
         
            self.similarity2=kmeans_test.shroom_predict(src,tgt,hyp, self.selected_option.get())
            self.after(5000, self.hide_progress_bar2)  
        else:
            sentence1, sentence2 = encode_pair([hyp, tgt])
            similarity = model(sentence1,sentence2)
            self.predicted_label = "Not Hallucination" if similarity.item() > 0.7 else "Hallucination"
            self.similarity=similarity.item()
            self.after(2000, self.hide_progress_bar)  





    def center_window(self):
        # Get window dimensions
        window_width = 400  # Width of the window
        window_height = 600  # Height of the window

        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate x and y coordinates for the Tk root window to be centered
        x = (screen_width / 2) - (window_width / 2)
        y = (screen_height / 2) - (window_height / 2)

        # Set the dimensions and position of the window
        self.geometry(f'{window_width}x{window_height}+{int(x)}+{int(y)}')
        # # Configure row/column weights to make text entry stretch
        # self.rowconfigure(1, weight=1)
        # self.columnconfigure(0, weight=1)

        # # Category selector dropdown
        # self.category_label = tk.Label(self, text="Category:")
        # self.category_label.grid(row=2, column=0, padx=10, pady=5)
        # self.category_var = tk.StringVar()
        # self.category_dropdown = ttk.Combobox(self, textvariable=self.category_var, values=["DM", "MT", "PG"])
        # self.category_dropdown.grid(row=2, column=1, padx=10, pady=5)

        # # Textboxes
        # self.target_label = tk.Label(self, text="Target:")
        # self.target_label.grid(row=3, column=0, padx=10, pady=5)
        # self.target_entry = tk.Entry(self)
        # self.target_entry.grid(row=3, column=1, padx=10, pady=5)

        # self.source_label = tk.Label(self, text="Source:")
        # self.source_label.grid(row=4, column=0, padx=10, pady=5)
        # self.source_entry = tk.Entry(self)
        # self.source_entry.grid(row=4, column=1, padx=10, pady=5)

        # # Button
        # self.evaluate_button = tk.Button(self, text="Evaluate", command=self.on_evaluate_click)
        # self.evaluate_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        # # Scale/Indicator element
        # self.probability_label = tk.Label(self, text="Probability:")
        # self.probability_label.grid(row=6, column=0, padx=10, pady=5)
        # self.probability_scale = ttk.Scale(self, from_=0, to=1, orient="horizontal")
        # self.probability_scale.grid(row=6, column=1, padx=10, pady=5)

    def on_evaluate_click(self):
        print("Evaluation clicked")

if __name__ == "__main__":
    app = CardEvaluationApp()
    app.mainloop()
