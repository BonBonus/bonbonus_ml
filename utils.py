from dataclasses import dataclass
import ast
import numpy as np

@dataclass
class ServiceParams:
    max_number_purchase: int
    bonus_type: list
    merchantName2id: map
    bonusName2id: map

def vectorize_for_bonus_predict(row, merchant_id, merchants_db, params: ServiceParams, train=True):
    vector = [row.age]
    if type(row.purchase_history) is str:
        purchase_history = ast.literal_eval(row.purchase_history)
    else:
        purchase_history = row.purchase_history
    rating = row.rating
    if train:
        merchant_id = row.current_merchant_id
        bonus_to_offer = row.bonus_to_offer
    current_merchant = merchants_db.iloc[merchant_id]
    current_merchant_mcc = current_merchant.merchant_mcc
    if type(current_merchant.bonuses_available) is str:
        bonuses_available = ast.literal_eval(current_merchant.bonuses_available)
    else:
        bonuses_available = current_merchant.bonuses_available
    for i in range(0, params.max_number_purchase):
        if i < len(purchase_history):
            vector.append(params.merchantName2id[purchase_history[i][0]]+1)
            vector.append(purchase_history[i][1])
        else:
            vector.extend([0, 0.0])
    vector.append(params.merchantName2id[current_merchant_mcc]+1)
    for i in range(0, len(params.bonus_type)):
        if i < len(bonuses_available):
            vector.append(params.bonusName2id[bonuses_available[i]]+1)
        else:
            vector.append(0)
    vector.append(rating)
    if train:
        if bonus_to_offer:
            vector.append(params.bonusName2id[bonus_to_offer]+1)
        else:
            vector.append(0)
    return vector

def vectorize_for_neighbours_predict(row, params: ServiceParams):
    vector = [row.age]
    if type(row.purchase_history) is str:
        purchase_history = ast.literal_eval(row.purchase_history)
    else:
        purchase_history = row.purchase_history
    rating = row.rating
    for i in range(0, params.max_number_purchase):
        if i < len(purchase_history):
            vector.append(params.merchantName2id[purchase_history[i][0]]+1)
            vector.append(purchase_history[i][1])
        else:
            vector.extend([0, 0.0])
    vector.append(rating)
    return vector

def predict_bonus(client_id, store_id, model, client_database, merchant_database, service_params: ServiceParams) -> str:
    if client_id not in client_database.index:
        return ''
    client_record = client_database.iloc[client_id]
    client_vector = vectorize_for_bonus_predict(client_record, store_id, merchant_database, service_params, train=False)
    client_vector = np.array(client_vector).astype(np.float32)
    prediction = model.predict_proba(client_vector.reshape(1,-1)).argmax()-1
    if prediction >= 0:
        return service_params.bonus_type[prediction]
    else:
        return ''
    
def predict_neighbours(client_id, model, client_database, service_params: ServiceParams) -> list:
    if client_id not in client_database.index:
        return ''
    client_record = client_database.iloc[client_id]
    client_vector = vectorize_for_neighbours_predict(client_record, service_params)
    client_vector = np.array(client_vector).astype(np.float32)
    vectors =  np.array(client_database.apply(vectorize_for_neighbours_predict, params=service_params, axis=1).to_list())
    X = vectors.astype(np.float32)
    model.fit(X)
    distances, indices = model.kneighbors(client_vector.reshape(1,-1))
    distances = distances[0]
    return indices[0][1:4], distances