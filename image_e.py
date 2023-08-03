with open('C:/Users/dvice/Desktop/Proyectos/FINAL PROJECT/anyone_data.json') as file:
    # Load the JSON data into a Python dictionary
    data_anyone = json.load(file)


info=[]
data_anyone=data_anyone[39568:]
for element in data_anyone:  
    response = requests.get(element['image'])
    #print(element['url'])
    #print(response.status_code)
    if response.status_code == 200:
      image = Image.open(BytesIO(response.content))
      image_name=str(element['sku'])+'.jpg'
      #print(image_name)
      info.append([element['sku'], image_name])
      new_width = 250
      new_height = 250
      image=image.convert('RGB')
      image = image.resize((new_width, new_height))
      image.save("C:/Users/dvice/Desktop/Proyectos/FINAL PROJECT/imagenes/"+image_name)
info=pd.DataFrame(info)
info.to_csv('data_image.csv', index=False, header=True)