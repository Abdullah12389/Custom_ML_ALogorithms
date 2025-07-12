<!DOCTYPE html>
<html>

<body>

  <h1>Complete Machine Learning and Transformer Implementation</h1>

  <div class="section">
    <h2>Description</h2>
    <p>
      This project is a complete manual implementation of machine learning models and transformer architecture using only <strong>NumPy</strong> and low-level <strong>TensorFlow</strong> APIs.
      No shortcuts, no scikit-learn pipelines, no high-level Keras layers. Just pure code to understand everything from scratch.
    </p>
  </div>

  <div class="section">
    <h2>Models Implemented</h2>
    <ul>
      <li>Linear Regression</li>
      <li>Logistic Regression</li>
      <li>K-Means Clustering</li>
      <li>Sophisticated Gradient Descent</li>
      <li>Decision Tree (Classification & Regression)</li>
      <li>Random Forest</li>
      <li>Gradient Boosting</li>
      <li><strong>Transformer Architecture:</strong>
        <ul>
          <li>Multi-Head Attention</li>
          <li>Positional Encoding</li>
          <li>Encoder & Decoder Layer</li>
          <li>Feedforward Network</li>
          <li>Masked & Cross Attention</li>
          <li>Softmax, dot product, reshape, concat (manual)</li>
          <li>Training with dummy data</li>
        </ul>
      </li>
    </ul>
  </div>

  <div class="section">
    <h2>Technologies Used</h2>
    <ul>
      <li>Python 3</li>
      <li>NumPy</li>
      <li>TensorFlow (Low-level APIs)</li>
    </ul>
  </div>

  <div class="section">
    <h2>How to Run</h2>
    <ol>
      <li>Define input tensors using <code>tf.convert_to_tensor()</code></li>
      <li>Create encoder and decoder layers using custom classes</li>
      <li>Connect them using <code>tf.keras.Model</code></li>
      <li>Compile with loss and optimizer</li>
      <li>Train using <code>model.fit()</code></li>
    </ol>
  </div>

  <div class="section">
    <h2>Installation</h2>
    <pre><code>pip install numpy tensorflow</code></pre>
  </div>

  <div class="section">
    <h2>Purpose</h2>
    <p>
      This project is built for <strong>deep learning and experimentation</strong> purposes.
      Every operation from reshaping, matrix multiplication, masking, and attention was manually built to help master the internal mechanics of ML and transformers.
    </p>
  </div>

</body>
</html>