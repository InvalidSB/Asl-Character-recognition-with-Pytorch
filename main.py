
import streamlit as st
import torch
import torchvision.transforms as transforms
import PIL.Image as Image


classes = ['A', 'B', "C", "D", "Delete", "E", "F", "G", "H", "I", "J", "K", "L", "M",
           "N", "Nothinh", "O", "P", "Q", "R", "S", "Space", "T", "U", "V", "W", "X", "Y", "Z"]
model = torch.load('./best_model.pth')

mean = [0.5159, 0.4969, 0.5109]
std = [0.1986, 0.2288, 0.2392]
image_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])


def process_image(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    return (classes[predicted.item()])


st.set_page_config(page_title="ASL character recognition...",
                   page_icon=":camera:", layout="centered")
st.title("ASL Character recognition application")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if file:
    st.image(file, caption='Uploaded Image.', width=200)
    if st.button("Submit"):
        result = process_image(model, image_transforms, file, classes)
        st.success("The image has been recognized successfully.")
        st.markdown('<h1 style="font-size: 50px; text-align: center">Result :</h1>',
                    unsafe_allow_html=True)
        Result_text = '<p style="font-family:Courier; color:yellow;font-weight:bold; font-size: 100px; text-align: center">{}</p>'
        st.markdown(Result_text.format(result), unsafe_allow_html=True)
