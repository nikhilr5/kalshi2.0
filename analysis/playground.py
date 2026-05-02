from utility import AnalysisUtils
u = AnalysisUtils()


iv = u.implied_vol_binary(.51, 78461, 78500, 0.016540, r=0.043)
print(iv) #.05

iv = u.implied_vol_binary(.48, 78461, 78500, 0.016540, r=0.043)
print(iv) #.81