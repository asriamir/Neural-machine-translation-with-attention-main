
# Neural Machine Translation with Attention

This project implements a Neural Machine Translation (NMT) model with an attention mechanism to translate human-readable dates into machine-readable ISO date format. The model is built using TensorFlow and Keras.

---

## Overview

The NMT model with attention takes as input dates in various human-readable formats (e.g., `3 May 1979`, `5 Apr 09`) and converts them to ISO format (`YYYY-MM-DD`). 

Key highlights:
- Uses a Bi-Directional LSTM for encoding the input sequence.
- Incorporates an attention mechanism for improved performance.
- Employs a post-attention LSTM for decoding.

---

## Features

- **Customizable input formats:** Includes multiple date formats for training.
- **One-step attention mechanism:** Focuses on relevant input sequences for better decoding.
- **Visualization:** Attention maps to interpret the model’s focus during translation.

---

## Dataset

The dataset is generated using the `Faker` library to simulate human-readable date strings. The `Babel` library is used to format dates.

- Example input: `3 May 1979`
- Example output: `1979-05-03`

---

## Setup and Usage

### Prerequisites

- Python >= 3.7
- TensorFlow/Keras
- `Faker`
- `Babel`
- `numpy`
- `matplotlib`
- `tqdm`

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Neural-machine-translation-with-attention
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `nmt_utils.py` file is in the same directory.

---

## Model Architecture

The architecture consists of:
- **Encoder:** A Bidirectional LSTM to process the input sequence.
- **Attention Mechanism:** To compute context vectors based on the encoder's outputs.
- **Decoder:** A post-attention LSTM to generate the target sequence step-by-step.
- **Dense Output Layer:** Applies a softmax activation for final predictions.

---

## Training the Model

1. **Dataset Generation:**

   Run `load_dataset` to generate a dataset of size `m`:

   ```python
   dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m=10000)
   ```

2. **Preprocess Data:**

   ```python
   X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx=30, Ty=10)
   ```

3. **Compile and Train the Model:**

   ```python
   opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
   model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
   model.fit([Xoh, s0, c0], outputs, epochs=10, batch_size=64)
   ```

---

## Testing the Model

To evaluate the model on new examples:

```python
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007']
for example in EXAMPLES:
    run_example(model, human_vocab, inv_machine_vocab, example)
```

---

## Visualizing Attention

To visualize the attention mechanism:

```python
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num=7, n_s=64)
```

---

## Contributing

Contributions are welcome! Please fork this repository and create a pull request with your changes.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [Faker Library](https://faker.readthedocs.io/)
- [Babel Library](http://babel.pocoo.org/)
- [Keras Documentation](https://keras.io/)
