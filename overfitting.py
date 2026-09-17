import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_excel("spreadsheet.xlsx")
X = df.drop(columns=["gia"])
y = df["gia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Tổng số mẫu: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")

mo_hinh_overfit = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    LinearRegression()
)
mo_hinh_overfit.fit(X_train, y_train)

so_dac_trung = mo_hinh_overfit.named_steps["polynomialfeatures"].n_output_features_
print(f"\nMô hình overfit (bậc 3)")
print(f"Số đặc trưng sau khi sinh đa thức: {so_dac_trung} (mẫu train: {len(X_train)})")

y_pred_train = mo_hinh_overfit.predict(X_train)
y_pred_test = mo_hinh_overfit.predict(X_test)

print(f"  R2 train:  {r2_score(y_train, y_pred_train):.4f}")
print(f"  R2 test:   {r2_score(y_test, y_pred_test):.4f}")
print(f"  MAE train: {mean_absolute_error(y_train, y_pred_train):,.0f} VND")
print(f"  MAE test:  {mean_absolute_error(y_test, y_pred_test):,.0f} VND")

print("\nCách 1: sai số train vs test theo bậc đa thức")
for bac in [1, 2, 3, 4]:
    m = make_pipeline(PolynomialFeatures(bac, include_bias=False),
                       StandardScaler(), LinearRegression())
    m.fit(X_train, y_train)
    n_feat = m.named_steps["polynomialfeatures"].n_output_features_
    r2_tr = r2_score(y_train, m.predict(X_train))
    r2_te = r2_score(y_test, m.predict(X_test))
    mae_tr = mean_absolute_error(y_train, m.predict(X_train))
    mae_te = mean_absolute_error(y_test, m.predict(X_test))
    print(f"  Bậc {bac} ({n_feat} đặc trưng)")
    print(f"    R2  train/test:  {r2_tr:.4f} / {r2_te:.4f}")
    print(f"    MAE train/test:  {mae_tr:,.0f} / {mae_te:,.0f} VND")

print("\nCách 2: validation set")
X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)
print(f"  Train: {len(X_tr2)} | Validation: {len(X_val)} | Test: {len(X_test)}")

bac_tot_nhat, r2_tot_nhat = None, -np.inf
for bac in [1, 2, 3]:
    m = make_pipeline(PolynomialFeatures(bac, include_bias=False),
                       StandardScaler(), LinearRegression())
    m.fit(X_tr2, y_tr2)
    r2_tr = r2_score(y_tr2, m.predict(X_tr2))
    r2_val = r2_score(y_val, m.predict(X_val))
    print(f"  Bậc {bac}: R2 train={r2_tr:.4f} | R2 validation={r2_val:.4f}")
    if r2_val > r2_tot_nhat:
        r2_tot_nhat, bac_tot_nhat = r2_val, bac
print(f"  => Chọn bậc {bac_tot_nhat} vì có R2 trên validation cao nhất")

print("\nCách 3: cross-validation (5-fold)")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for bac in [1, 2, 3]:
    m = make_pipeline(PolynomialFeatures(bac, include_bias=False),
                       StandardScaler(), LinearRegression())
    scores = cross_val_score(m, X_train, y_train, cv=kf, scoring="r2")
    print(f"  Bậc {bac}: R2 trung bình = {scores.mean():.4f} (độ lệch chuẩn = {scores.std():.4f})")
    print(f"    từng fold: {np.round(scores, 3).tolist()}")

print("\nCách 4: Lasso regularization trên mô hình overfit (bậc 3)")
print("  Alpha được chọn bằng cross-validation trên tập train, không dùng tập test")
mo_hinh_lasso = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    LassoCV(cv=5, random_state=42, max_iter=20000)
)
mo_hinh_lasso.fit(X_train, y_train)
so_dac_trung_ve_0 = (mo_hinh_lasso.named_steps["lassocv"].coef_ == 0).sum()
alpha_tot_nhat = mo_hinh_lasso.named_steps["lassocv"].alpha_

print(f"  Alpha tốt nhất (chọn bởi CV): {alpha_tot_nhat:.4f}")
print(f"  R2 train:  {r2_score(y_train, mo_hinh_lasso.predict(X_train)):.4f}")
print(f"  R2 test:   {r2_score(y_test, mo_hinh_lasso.predict(X_test)):.4f}")
print(f"  MAE test:  {mean_absolute_error(y_test, mo_hinh_lasso.predict(X_test)):,.0f} VND")
print(f"  Số đặc trưng bị triệt tiêu về 0: {so_dac_trung_ve_0} / {so_dac_trung}")

final_model = mo_hinh_lasso

ket_qua = pd.DataFrame({
    "Gia_thuc_te": y_test,
    "Gia_du_doan": final_model.predict(X_test).round(0)
})
print("\n5 căn nhà đầu tiên trong tập kiểm thử")
print(ket_qua.head(5).to_string(index=False))