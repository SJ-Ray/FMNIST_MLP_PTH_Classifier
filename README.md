# Fashion Mnist Multi-Layer Perceptron Classifier
This notebook demonstrate the use of PyTorch to create a Multi-Layer Perceptron for Image Classification on Fashion Mnist Dataset.

I have used Fashion Mnist Dataset which contains:
<ul>
<li>60K training datapoint.</li>
<li>10K test datapoints.</li>
<li>10 Categories.</li>
</ul>

<table>
<thead>
<tr>
<th>Label</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>T-shirt/top</td>
</tr>
<tr>
<td>1</td>
<td>Trouser</td>
</tr>
<tr>
<td>2</td>
<td>Pullover</td>
</tr>
<tr>
<td>3</td>
<td>Dress</td>
</tr>
<tr>
<td>4</td>
<td>Coat</td>
</tr>
<tr>
<td>5</td>
<td>Sandal</td>
</tr>
<tr>
<td>6</td>
<td>Shirt</td>
</tr>
<tr>
<td>7</td>
<td>Sneaker</td>
</tr>
<tr>
<td>8</td>
<td>Bag</td>
</tr>
<tr>
<td>9</td>
<td>Ankle boot</td>
</tr>
</tbody>
</table>

The model used is a Pytorch Sequential model with the following architecture.
<br>
<pre>
Sequential(
    (0): Linear(in_features=784, out_features=128, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2)
    (3): Linear(in_features=128, out_features=64, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.2)
    (6): Linear(in_features=64, out_features=10, bias=True)
    (7): LogSoftmax()
  )
</pre>

 Classification report on the test data set:
<br> 

 <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>f1-score</th>
      <th>precision</th>
      <th>recall</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>T-shirt/top</td>
      <td>0.800395</td>
      <td>0.821501</td>
      <td>0.780347</td>
      <td>519.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Trouser</td>
      <td>0.966732</td>
      <td>0.951830</td>
      <td>0.982107</td>
      <td>503.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pullover</td>
      <td>0.743383</td>
      <td>0.674322</td>
      <td>0.828205</td>
      <td>390.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dress</td>
      <td>0.830612</td>
      <td>0.814000</td>
      <td>0.847917</td>
      <td>480.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Coat</td>
      <td>0.751342</td>
      <td>0.876827</td>
      <td>0.657277</td>
      <td>639.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sandal</td>
      <td>0.931206</td>
      <td>0.906796</td>
      <td>0.956967</td>
      <td>488.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Shirt</td>
      <td>0.576375</td>
      <td>0.546332</td>
      <td>0.609914</td>
      <td>464.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sneaker</td>
      <td>0.921105</td>
      <td>0.934000</td>
      <td>0.908560</td>
      <td>514.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bag</td>
      <td>0.959402</td>
      <td>0.947257</td>
      <td>0.971861</td>
      <td>462.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ankle boot</td>
      <td>0.945489</td>
      <td>0.961759</td>
      <td>0.929760</td>
      <td>541.0</td>
    </tr>
    <tr>
      <th>micro avg</th>
      <td></td>
      <td>0.843600</td>
      <td>0.843600</td>
      <td>0.843600</td>
      <td>5000.0</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td></td>
      <td>0.842604</td>
      <td>0.843462</td>
      <td>0.847292</td>
      <td>5000.0</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td></td>
      <td>0.844092</td>
      <td>0.850633</td>
      <td>0.843600</td>
      <td>5000.0</td>
    </tr>
  </tbody>
</table>
