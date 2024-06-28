def download_images(predictions, key, image_limit=1):
    from pygoogle_image import image as pi
    import os
    for product_name in predictions[key].values():
        # 定義圖片存放的根目錄
        images_dir = "images"
        # 生成產品名稱對應的資料夾路徑
        product_dir = os.path.join(images_dir, product_name.replace(' ', '_')) # 資料夾名稱空白會自動補'_'
        # 檢查資料夾是否已經存在
        if not os.path.exists(product_dir):
            # 不存在就使用 pygoogle_image 下載圖片
            try:
                pi.download(product_name, limit=image_limit)
                # os.chdir("..")
                print(f"成功下載產品： {product_name}")
            except Exception as e:
                print(f"發生下載錯誤的產品： {product_name} {e}")
        else:
            print(f"產品 {product_name} 已經存在")

def get_recommendations(user_id_data=None, data_path='../datasets/'):

    # 載入套件
    import pandas as pd
    import pickle
    from datetime import datetime
    import xgboost as xgb
    from f1optimization_faron import get_best_prediction

    # 獲取當前時間(計時用)
    # 計時開始
    start_time = datetime.now()

    # 獲取當前時間(計算日期用)
    now = datetime.now()
    # 格式化日期和時間
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    today = int(dt_string.split("/")[0])

    # 從用戶端獲取數據
    user_id = int(user_id_data['user_id'])  # 用戶ID
    order_hour_of_day = int(dt_string.split(" ")[1].split(":")[0])  # 當前小時
    order_dow = datetime.today().weekday()  # 當前星期幾

    # 讀取用戶最後訂單日期(預設為部屬日期 6/14) 的 pkl 檔
    ulp = pd.read_pickle(data_path + "user_last_purchase.pkl")
    # 檢查用戶是否為新用戶
    if user_id not in ulp['user_id'].values:
        # 讀取基於小時和星期幾銷售前 10 的產品 pkl 檔
        top = pd.read_pickle(data_path + 'top10_products.pkl')
        top_products = top[(top['order_dow'] == order_dow) & (top['order_hour_of_day'] == order_hour_of_day)]['product_name'].values.tolist() # list 型態
        top_products = {i: value for i, value in enumerate(top_products)} # 轉換為字典型態
        predictions = {'top': top_products}

        # 使用 pygoogle_image 下載圖片
        download_images(predictions, key='top', image_limit=1)

        # 計時結束
        end_time = datetime.now()
        difference = end_time - start_time
        time = "{}".format(difference)
        return predictions, time


    # 判斷時否為原用戶
    if user_id in ulp['user_id'].values:

        user_last_order_date = ulp[ulp['user_id'] == user_id]['date'].values.tolist()[0] # 獲取預設最後訂單日期
        days_since_prior_order = today - int(user_last_order_date.split('-')[-1]) # 計算距離上次購買過了幾天
        # 用戶最後訂單日期和計算的天數
        print("用戶最後訂單日期:", user_last_order_date)
        print("距離上次購買間隔天數:", days_since_prior_order)


        # 依據以上資訊創建特徵
        misc_hour_order_rate = pd.read_pickle(data_path + "hour_order_rate.pkl") # 每小時的訂購率
        misc_day_order_rate = pd.read_pickle(data_path + "day_order_rate.pkl") # 每星期幾的訂購率
        misc_product_days_since_prior_order_order_rate = pd.read_pickle(data_path + "product_days_since_prior_order_order_rate.pkl") # 各個產品不同間隔天數的訂購率
        misc_user_days_since_prior_order_order_rate = pd.read_pickle(data_path + "user_days_since_prior_order_order_rate.pkl") # 各個用戶不同間隔天數的訂購率
        misc_uxp_days_since_prior_order_order_rate = pd.read_pickle(data_path + "uxp_days_since_prior_order_order_rate.pkl") # 各個用戶對於各個產品不同間隔天數的訂購率


        # 其餘特徵已經在特徵工程中完成，因此其他特徵直接讀取 data.h5 即可
        path = data_path + 'train_test_data.h5'
        train_data = pd.read_hdf(path, key='train')
        test_data = pd.read_hdf(path, key='test')
        # 刪除 要根據當下情況判斷的特徵 以及 模型不需要的特徵
        train_data.drop(columns=['order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order',
                                'misc_hour_order_rate', 'misc_day_order_rate',
                                'misc_product_days_since_prior_order_order_rate',
                                'misc_user_days_since_prior_order_order_rate',
                                'misc_uxp_days_since_prior_order_order_rate', 'reordered'], inplace=True)

        test_data.drop(columns=['order_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order',
                                'misc_hour_order_rate', 'misc_day_order_rate',
                                'misc_product_days_since_prior_order_order_rate',
                                'misc_user_days_since_prior_order_order_rate',
                                'misc_uxp_days_since_prior_order_order_rate'], inplace=True)
        # 合併成一個完整原用戶的數據特徵集
        all_user_data = pd.concat([train_data, test_data], axis=0)
        # 抓取該用戶的特徵數據
        featurized_data = all_user_data[all_user_data['user_id'] == user_id]
        # 新增間隔時間的特徵
        featurized_data = featurized_data.copy()
        featurized_data['days_since_prior_order'] = days_since_prior_order


        # 根據用戶ID、時間和星期收集特徵
        hour_rate = misc_hour_order_rate[misc_hour_order_rate['order_hour_of_day'] == order_hour_of_day]
        day_rate = misc_day_order_rate[misc_day_order_rate['order_dow'] == order_dow]
        p_days_rate = misc_product_days_since_prior_order_order_rate[misc_product_days_since_prior_order_order_rate['days_since_prior_order'] == days_since_prior_order]
        u_days_rate = misc_user_days_since_prior_order_order_rate[(misc_user_days_since_prior_order_order_rate['user_id'] == user_id) & (misc_user_days_since_prior_order_order_rate['days_since_prior_order'] == days_since_prior_order)]
        uxp_days_rate = misc_uxp_days_since_prior_order_order_rate[(misc_uxp_days_since_prior_order_order_rate['user_id'] == user_id) & (misc_uxp_days_since_prior_order_order_rate['days_since_prior_order'] == days_since_prior_order)]
        # 注意：並不是每個用戶和產品都會有所有時間、間隔天數的數據，因此我們需要給定預設值
        # 處理空的特徵數據
        if p_days_rate.empty:
            p_days_rate = pd.DataFrame(columns=p_days_rate.columns)
            products_x = pd.read_pickle(data_path + 'products_id_name.pkl')
            p_days_rate['product_id'] = p_days_rate['product_id']
            p_days_rate['days_since_prior_order'] = float(days_since_prior_order)
            p_days_rate['misc_product_days_since_prior_order_order_rate'] = float(0)
            del products_x
        if u_days_rate.empty:
            u_days_rate = pd.DataFrame(columns=u_days_rate.columns)
            df2 = {'user_id': user_id, 'days_since_prior_order': float(days_since_prior_order), 'misc_user_days_since_prior_order_order_rate': float(0)}
            u_days_rate = pd.concat([u_days_rate, pd.DataFrame([df2])], ignore_index=True)
            del df2
        if uxp_days_rate.empty:
            uxp_days_rate = pd.DataFrame(columns=misc_uxp_days_since_prior_order_order_rate.columns)
            products_x = pd.read_pickle(data_path + 'products_id_name.pkl')
            uxp_days_rate['product_id'] = products_x['product_id']
            uxp_days_rate['user_id'] = user_id
            uxp_days_rate['days_since_prior_order'] = float(days_since_prior_order)
            uxp_days_rate['misc_uxp_days_since_prior_order_order_rate'] = float(0)
            del products_x

        # 合併特徵數據
        featurized_data = pd.merge(featurized_data, hour_rate, on='product_id')
        featurized_data = pd.merge(featurized_data, day_rate, on='product_id')
        featurized_data = pd.merge(featurized_data, p_days_rate, on=['product_id', 'days_since_prior_order'])
        featurized_data = pd.merge(featurized_data, u_days_rate, on=['user_id', 'days_since_prior_order'])
        featurized_data = pd.merge(featurized_data, uxp_days_rate, on=['user_id', 'product_id', 'days_since_prior_order'])

        # 將時間轉換為類別變數
        def hour_to_categorical(time):
            if 6 <= time < 12:  # 6 AM 到 11 AM
                return 0
            elif 12 <= time < 17:  # 12 PM 到 4 PM
                return 1
            elif 17 <= time < 21:  # 5 PM 到 8 PM
                return 2
            else:  # 9 PM 到 5 AM
                    return 3
        featurized_data['order_hour_of_day'] = featurized_data['order_hour_of_day'].apply(hour_to_categorical)

        # 預測的特徵要與模型的特徵順序一致
        expected_features = ['uxp_order_rate', 'uxp_order_reorder_ratio', 'uxp_avg_position', 'uxp_orders_since_last',
                            'uxp_max_streak', 'user_reorder_rate', 'user_unique_products', 'user_total_products',
                            'user_avg_cart_size', 'user_avg_days_between_orders', 'user_reordered_products_ratio',
                            'product_reorder_ratio', 'product_avg_pos_incart', 'product_reduced_feat_1',
                            'product_reduced_feat_2', 'product_reduced_feat_3', 'product_aisle_reorder_rate',
                            'product_department_reorder_rate', 'uxp_product_unique_customers', 'uxp_product_one_shot_ratio',
                            'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'misc_hour_order_rate',
                            'misc_day_order_rate', 'misc_product_days_since_prior_order_order_rate',
                            'misc_user_days_since_prior_order_order_rate', 'misc_uxp_days_since_prior_order_order_rate']
        data = featurized_data[expected_features]

        # 加載模型並進行預測
        with open(data_path + "xgb_v1.pkl", "rb") as f:
            model = pickle.load(f)
        dtest = xgb.DMatrix(data)
        ypred = model.predict(dtest)
        del dtest, model

        # 獲取最有可能的產品集
        recommended_products = get_best_prediction(featurized_data['product_id'].tolist(), ypred.tolist(), None, showThreshold=False) # get_best_prediction會回傳product_id且用' '隔開的字串
        recommended_products = recommended_products.replace("None", "")
        recommended_products = list(map(int, recommended_products.split())) # 我們的 product_id 是 int 型態的，所以需要轉成 int 的 list
        products_x = pd.read_pickle(data_path + 'products_id_name.pkl')
        recommended_products_df = products_x[products_x['product_id'].isin(recommended_products)]
        recommended_products = {i: value for i, value in enumerate(recommended_products_df['product_name'])} # 轉成字典
        predictions = {'recommend': recommended_products}

        # 使用 pygoogle_image 下載圖片
        download_images(predictions, key='recommend', image_limit=1)

        # 計時結束
        end_time = datetime.now()
        difference = end_time - start_time
        time = "{}".format(difference)

        return predictions, time