# Language Identification Project

### problem:
 what is the language of this sentence?

### solution
 There are alot of ways to takkle this problem using various machine learning techniques,
 including Linear Regression, Support Vector Machines, Deep Neural Networks (LSTMs or GRUs), and transformers.
 So for the purpose of learning i tried them all and here is the results in the follwing table.
 <table>
   <thead>
     <tr>
       <th>Algorithm</th>
       <th>accuracy</th>
     </tr>
   </thead>
   <tbody>
     <tr>
       <td>Linear regression</td>
       <td>0.939</td>
     </tr>
     <tr>
       <td>Suport Vector Classifier</td>
       <td>0.937</td>
     </tr>
     <tr>
       <td>RNN (bidirectional GRU)</td>
       <td>.92</td>
     </tr>
     <tr>
        <td>fine-tuning  ‘xlm-roberta-base’ Transformer</td>
       <td>0.99</td>
     </tr>
   </tbody>
 </table>

 ### Conclusion
   We got the best answer by fine-tuning ‘xlm-roberta-base’ Transformer with accuracy 99% in both test and train sets
   hence the name "Attention is all you need" :).
