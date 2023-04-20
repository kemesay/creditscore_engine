from flask import Flask, request, jsonify
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from torch_utils import preprocess, result
app = Flask(__name__)


#######################################################################################################################
#######################################################################################################################
ALLOWED_EXTENSIONS = {'csv', 'xls'}

def allowed_file(filename):
       return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
       if request.method == 'POST':
              file = request.files.get ('file')
              if file is None or file.filename =="":
                 return jsonify({'error':'no file'})
              
              if not  allowed_file(file.filename):
                     return jsonify({'error':'format not supported'})
              
              # try:
              # dt12 = pd.read_csv('indiv.csv')
              dt12 = pd.read_csv(file)
              # tensor = preprocess(img_bytes)
              # prediction = predict(tensor)
              # data = {'prediction':prediction.item, 'class_name':str(prediction.item())}
              # return jsonify(data)
              
              ##########################################################################################################################
              ##########################################################################################################################

              num_col = ['2022-06-24','2022-06-25','2022-06-27','2022-06-28','2022-06-29',	'2022-06-30', '2022-07-01','2022-07-02', '2022-07-04',
                            '2022-07-05','2022-07-06', '2022-07-07','2022-07-08','2022-07-11','2022-07-12','2022-07-13',	'2022-07-14',	'2022-07-15',
                            '2022-07-16','2022-07-18',  '2022-07-19','2022-07-20','2022-07-21', '2022-07-22', '2022-07-23', '2022-07-25', '2022-07-26',	'2022-07-27',	'2022-07-28',	'2022-07-29',
                            '2022-07-30', '2022-08-01', '2022-08-02', '2022-08-03', '2022-08-04', '2022-08-05', '2022-08-06','2022-08-08',	'2022-08-09', '2022-08-10', '2022-08-11',
                            '2022-08-12', '2022-08-13', '2022-08-15', '2022-08-16', '2022-08-17', '2022-08-18', '2022-08-19', '2022-08-20', '2022-08-22', '2022-08-23',	'2022-08-24',	
                            '2022-08-25', '2022-08-26', '2022-08-27', '2022-08-29', '2022-08-30', '2022-08-31', '2022-09-01', '2022-09-02',	'2022-09-03', '2022-09-05',	'2022-09-06', '2022-09-07',
                            '2022-09-08', '2022-09-09', '2022-09-10','2022-09-12', '2022-09-13', '2022-09-14', '2022-09-15',	 '2022-09-16',	'2022-09-17', '2022-09-19', '2022-09-20',
                            '2022-09-21', '2022-09-22', '2022-09-23','2022-09-24','AGE', 'Av_monthly_income']
              cat_col = ['GENDER', 'Mar_status', 'Education_level', 'Occupation']
              label_col = 'target'

              
              data = pd.read_csv('mondaylast_data.csv')
              ####################################################################################################################################
              ####################################################################################################################################
              data = data.drop('CUSTOMER_ID', axis=1)
              data = data.loc[1:1705200, :]
              
              pca = PCA(n_components=2)
              X_pca = pca.fit_transform(data)
              
              kmeans = KMeans(n_clusters=2, n_init='auto')
              labels = kmeans.fit_predict(X_pca)
              data['labe'] = labels
              data = data.drop('target', axis=1)
              data = data.rename(columns={'labe': 'target'})
              
              #######################################################################################################################################
              dat = pd.read_csv('mondaylast_data.csv')
              dat = dat.drop('CUSTOMER_ID', axis=1)
              lab = 5
              dat = dat.iloc[[lab ]]
              dat.to_csv('indiv.csv', index=False)
              # dt12 = pd.read_csv('indiv.csv')
              
              #######################################################################################################################################
              # data = data.drop('Customer_ID', axis=1)
              dt = data.loc[3:11, :]

              dt = pd.concat([dt12, dt]).reset_index(drop=True)
              train_num, test_num, train_cat, test_cat, feature_sizes, ce, ne = preprocess(data, num_col, cat_col, label_col)


              the_model = torch.load("model12.pt")
              the_model.eval()
              return  result(the_model, dt, ce, ne)


              # except:
              # return jsonify({'error':'error during preedictions'})
   
   
                   

