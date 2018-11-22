
import pytesseract
from PIL import Image
import spacy
import plac
from pathlib import Path
import pandas as pd
from pandas import Series,DataFrame


spacy.prefer_gpu()

im = Image.open("D:\Programacao\Python\dados\ImagensProj\currculo-portugus-1-728.jpg")

Nome_Pessoa = []
Elementos_Ent = []


#CARREGAR O MODELO CRIADO E VERIFICAR SE O MESMO CONSEGUE ATENDER OS REQUISITOS

texto = pytesseract.image_to_string(im,lang="por")

print(texto)

#nlp2 = spacy.load('pt')
nlp2 = spacy.load('en_core_web_sm')

#improve with "Enumarate"
doc = nlp2(texto)


for  indice ,ent in enumerate( doc.ents):
    #print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    #print(ent.label_)
    if(ent.label_=='PERSON' and indice==0):
       print("->>>"+ent.text)
       Nome_Pessoa.append(ent.text)




#Carregando o modelo gabriel

#terminar essa frente ler a foto e rodar o treino

#carregando o modelo criado

plac.annotations(
output_dir = ("D:\Programacao\Python\dados\Treino-modelo","option","o",Path)
)

output_dir =Path("D:\Programacao\Python\dados\Treino-modelo")



nlp3 = spacy.load(output_dir)

doc2 = nlp3(texto)



for ind,ent2 in  enumerate(doc2.ents):

    if(ind != 0):
      print("indice-> " + str(ind))
      print("Variaveis coletadas do curriculo -> ",ent2.text,"Label-> ",ent2.label_)
      Elementos_Ent.append(ent2.text)


#Elementos dentro do DataFrame Enviar para a planilia
data = {"Skill":Elementos_Ent}
frame = pd.DataFrame(data)

print(frame)


frame.to_csv("D:\Programacao\Python\dados\saidas\Arquivo.csv",mode='a', header=True,sep=";")





