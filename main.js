const SeedRandom = require('seedrandom')(34);

let data = [
  { input: [0, 0], output: 0 },
  { input: [1, 0], output: 1 },
  { input: [0, 1], output: 1 },
  { input: [1, 1], output: 0 },
];

const weights = {
  i1_h1: SeedRandom(),
  i1_h2: SeedRandom(),
  i2_h1: SeedRandom(),
  i2_h2: SeedRandom(),
  h1_o1: SeedRandom(),
  h2_o1: SeedRandom(),
};

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
//Производное сигмоита
const derivative_sigmoid = (x) => {
  const fx = sigmoid(x);
  return fx * (1 - fx);
};

const NN = (i1, i2) => {
  const h1_input = weights.i1_h1 * i1 + weights.i2_h1 * i2;
  const h1 = sigmoid(h1_input);

  const h2_input = weights.i1_h2 * i1 + weights.i2_h2 * i2;
  const h2 = sigmoid(h2_input);

  const O1_input = weights.h1_o1 * h1 + weights.h2_o1 * h2;
  const o1 = sigmoid(O1_input);

  return o1;
};

const showResult = () => {
  data.forEach(({ input: [i1, i2], output: y }) => {
    console.log(`${i1} XOR ${i2} => ${NN(i1, i2)} expected ${y}`);
  });
};

showResult();

//Обучение
//функция ошибки
// (F(x) - y) ^ 2;

const train = () => {
  const weights_deltais = {
    i1_h1: 0,
    i1_h2: 0,
    i2_h1: 0,
    i2_h2: 0,
    h1_o1: 0,
    h2_o1: 0,
  };

  for (const {
    input: [i1, i2],
    output,
  } of data) {
    const h1_input = weights.i1_h1 * i1 + weights.i2_h1 * i2;
    const h1 = sigmoid(h1_input);

    const h2_input = weights.i1_h2 * i1 + weights.i2_h2 * i2;
    const h2 = sigmoid(h2_input);

    const O1_input = weights.h1_o1 * h1 + weights.h2_o1 * h2;
    const o1 = sigmoid(O1_input);

    const delta = output - o1;
    const o1_delta = delta * derivative_sigmoid(O1_input);

    weights_deltais.h1_o1 += h1 * o1_delta;
    weights_deltais.h2_o1 += h2 * o1_delta;

    const h1_delta = o1_delta * derivative_sigmoid(h1_input);
    const h2_delta = o1_delta * derivative_sigmoid(h2_input);

    weights_deltais.i1_h1 += i1 * h1_delta;
    weights_deltais.i2_h1 += i2 * h1_delta;

    weights_deltais.i1_h1 += i1 * h2_delta;
    weights_deltais.i2_h2 += i2 * h2_delta;
  }

  return weights_deltais;
};

const applyTrainUpdate = (deltas = train()) => {
  Object.keys(weights).forEach((key) => {
    weights[key] += deltas[key];
  });
};

applyTrainUpdate();
showResult();

console.log('_______________________');
