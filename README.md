
                    

# 🛢️ Crude Oil Price Prediction





## 🔍 Project Description
This project involves the construction of a neural network designed to predict crude oil prices using given raw material prices. The neural network was built using a Gated Recurrent Unit (GRU) with a window size of 20 and a hidden size of 64, spread across three layers. The code was written in Pytorch. 





The training set includes data from January 2018 to June 2023, while the test set includes data from July 2023 to June 2024. Additional data was utilized in the form of the CBOE Interest Rate 10 Year T No (^TNX) which was downloaded from yfinance. 





Raw materials were filtered by checking the coefficient through a linear regression attempt, allowing us to focus on raw materials that significantly influence predictions. The final model utilized a total of 7 raw materials, a batch size of 32, 80 epochs, and an Adam optimizer with a learning rate of 0.0001. The final model achieved a denormalized RMSE of 3.7941.





## ✨ Key Features
- ⚙️ **Neural Network**: GRU-based network with a window size of 20 and hidden size of 64
- 📚 **Data**: Utilizes data from January 2018 to June 2024, including additional data from CBOE Interest Rate 10 Year T No (^TNX)
- 📈 **Raw Materials**: Filters raw materials based on their impact on predictions
- 🧮 **Optimization**: Uses an Adam optimizer with a learning rate of 0.0001






## 🎉 Acknowledgments
We would like to express our gratitude to the open-source community for their resources and support. Special thanks to Korea University for providing a platform for this innovative project.
