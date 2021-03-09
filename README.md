# Types of Wine Classification using Multi-LayerPerceptronand Extreme Learning Machine <br>

## Data info
Use Wine Data Set from UCI : [data](https://archive.ics.uci.edu/ml/datasets/wine) <br>
In data set it has 178 row, 14 col and 14 attributes

1. Type มี 3 Class คือ 1, 2, 3 ของ ไวน์
2. Alcohol เป็นค่าแอลกอฮอล์ที่อยู่ในไวน์
3. Malic acid เป็นค่ากรดมาลิคที่อยู่ในผลไม้แต่ละชนิดและเป็นกรดที่ให้รสชาติเปรี้ยวของไวน์ 4. Ash เป็นเถ้าที่อยู่ในไวน์ ซึ่งเป็นสาร inorganic ที่เหลืออยู่จากการเผาไหม้ ปกติไม่เกิน 2.5
5. Alcalinity of ash เป็นค่าความเป็นด่างของเถ้าที่อยู่ในไวน์
6. Magnesium เป็นค่า แมกนีเซียม ที่อยู่ในไวน์
7. Total phenols
8. Flavanoids เป็นค่าสารพฤกษาเคมีที่พบได้ในผลไม้แต่ละชนิด
9. Nonflavanoid phenols เป็นค่าสารที่ไม่ใช่ Flavanoids ที่พบได้ในผลไม้แต่ละชนิด
10. Proanthocyanins เป็นสารที่ได้จากการสกัดเมล็ดและเปลือกขององุ่น (ผลไม้)
11. Color intensity เป็นค่าความเข้มสีของไวน์
12. Hue เป็นค่าสีความมืดหรือความสว่างของสีไวน์
13. OD280/OD315 of diluted wines เป็นค่าความเข้มข้นของโปรตีนที่ใช้กำหนดปริมาณโปรตีนของไวน์
14. Proline เป็นค่ากรดอะมิโนที่สำคัญ มีผลในการสร้างยีสต์ของไวน์

## Preprocessing
 * Check missing values if it missing replace with median data
 * Normalization data with Min-Max Normalization

## Splitting
 * Splitting Train, Test Data 70% and 30%

## Modeling
 * Use Attributes 2-14 for features class
 * Use Attribute 1 for labels class
 * Use Sigmoid for Activation function
 * Set Hidden node for 30, 40, 50, 60 and 70
 * Find Train and Test accuracy



