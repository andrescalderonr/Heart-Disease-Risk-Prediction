from model.inference import predict_proba

patient = [2.5, 1]  # ST Depression, Vessels
patient1 = [60, 300]

prob = predict_proba(patient)
prob2 = predict_proba(patient1)

print(f"Heart disease probability: {prob:.3f}")
print(f"Heart disease probability: {prob2:.3f}")
