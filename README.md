Install the required libraries using the following command 

  ```
    pip install -r requirements.txt
  ```


  1.  Check FP32 Accuracy
  ```
    python test.py
  ```

  2.  Quantize the FP32 trained model and save it
  ```
    python quantize.py
  ```

  3.  Test INT8 qunatized Model
  ```
    python test_quantize.py
  ```
