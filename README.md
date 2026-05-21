# DFU Web Frontend

এই project-টা তোমার thesis notebook-এর trained model use করে web-based prediction UI দেয়।

## Features

- image upload
- dropdown দিয়ে `MobileNetV2`, `ResNet50`, `LiteDFU-Net` model select
- selected model দিয়েই prediction
- `Normal` / `Ulcer` result
- notebook থেকে নেয়া threshold-based decision

## Project Structure

- `app.py` - Streamlit frontend + prediction logic
- `models/` - trained `.keras` files রাখার folder

## Model Files

`models/` folder-এ এই files রাখো:

- `best_MobileNetV2.keras`
- `best_ResNet50.keras`
- `best_LiteDFU-Net.keras`

তোমার notebook-এ model checkpoint save হচ্ছে:

```python
checkpoint_path = f"/kaggle/working/best_{model_name}.keras"
```

তাই Kaggle run শেষ হলে `/kaggle/working/` থেকে এই `.keras` files download করে
এই project-এর `models/` folder-এ রাখলেই হবে।

Notebook-এর saved thresholds:

- `MobileNetV2` -> `0.58`
- `ResNet50` -> `0.24`
- `LiteDFU-Net` -> `0.29`

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Browser-এ link open হলে image upload করে model select করে prediction নিতে পারবে।
# dfu-thesis
