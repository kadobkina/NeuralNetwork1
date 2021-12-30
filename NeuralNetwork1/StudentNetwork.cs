using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        // Входные веса
        public double[][,] weights;

        // Входные значения слоев
        public double[][] layerValues;

        // Матрица ошибки, вычисляется по какой-нибудь хитрой формуле
        public double[][] error;

        // Секундомер спортивный, завода «Агат», измеряет время пробегания стометровки, ну и время затраченное на обучение тоже умеет
        public Stopwatch stopWatch = new Stopwatch();

        /// <summary>
        ///  Конструктор сети с указанием структуры (количество слоёв и нейронов в них)
        /// </summary>
        /// <param name="structure">Массив с указанием нейронов на каждом слое (включая сенсорный)</param>
        public StudentNetwork(int[] structure)
        {
            weights = new double[structure.Length - 1][,];
            layerValues = new double[structure.Length][];
            error = new double[structure.Length][];

            for (int k = 0; k < structure.Length; k++)
            {
                error[k] = new double[structure[k]];
                layerValues[k] = new double[structure[k] + 1];
                layerValues[k][structure[k]] = 1;

                if (k == structure.Length - 1)
                    break;

                // количество весов = кол-во нейронов текущего слоя * кол-во нейронов следующего слоя
                int curLayer = structure[k] + 1, nextLayer = structure[k + 1];
                weights[k] = new double[curLayer, nextLayer];

                var r = new Random();
                for (int i = 0; i < curLayer; i++)
                    for (int j = 0; j < nextLayer; j++)
                        weights[k][i, j] = -1 + r.NextDouble() * (1 - (-1));
            }
        }

        // Проходим по сети
        private void Run(double[] input)
        {
            for (int j = 0; j < input.Length; j++)
                layerValues[0][j] = input[j];

            // активируем слои сети
            for (int layN = 1; layN < layerValues.GetLength(0); layN++)
            {
                for (int i = 0; i < weights[layN - 1].GetLength(1); i++)
                {
                    double sum = 0;
                    for (int j = 0; j < weights[layN - 1].GetLength(0); j++)
                        sum += layerValues[layN - 1][j] * weights[layN - 1][j, i];

                    // сигмоидальная функция
                    layerValues[layN][i] = 1 / (1 + Math.Exp(-sum));
                }
            }
        }

        // Метод обратного распространения ошибки
        private void BackPropagationMethod(double[] output)
        {
            // получаем значение ошибки на каждом решающем выходном нейроне
            for (var j = 0; j < output.Length; j++)
            {
                var curValue = layerValues[error.Length - 1][j];
                error.Last()[j] = curValue * (1 - curValue) * (output[j] - curValue);
            }
            // пересчитываем значение ошибки на остальных нейронах
            for (int i = error.Length - 2; i >= 1; i--)
                for (int j = 0; j < error[i].Length; j++)
                {
                    double sum = 0;
                    var curValue = layerValues[i][j];
                    for (int k = 0; k < error[i + 1].Length; k++)
                        sum += error[i + 1][k] * weights[i][j, k];

                    error[i][j] = curValue * (1 - curValue) * sum;
                }
            // меняем веса после пересчета ошибки
            for (int n = 0; n < weights.Length; n++)
            {
                for (int i = 0; i < weights[n].GetLength(0); i++)
                {
                    for (int j = 0; j < weights[n].GetLength(1); j++)
                    {
                        var dWeight = error[n + 1][j] * layerValues[n][i];
                        weights[n][i, j] += dWeight;
                    }
                }
            }
        }

        // Вычисленная суммарная квадратичная ошибка сети. Предполагается, что целевые выходы - 1 для верного, и 0 для остальных
        private double Error(double[] output)
        {
            double result = 0;

            for (int i = 0; i < output.Length; i++)
                result += Math.Pow(error.Last()[i], 2);

            return result;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iters = 1;
            while (sample.EstimatedError() > acceptableError)
            {
                Run(sample.input);
                BackPropagationMethod(sample.Output);
                ++iters;
            }
            return iters;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            //  Сначала надо сконструировать массивы входов и выходов
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

            //  Теперь массивы из samplesSet группируем в inputs и outputs
            for (int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].Output;
            }

            //  Текущий счётчик эпох
            int epoch_to_run = 0;
            // максимизируем ошибку
            double error = double.PositiveInfinity;

#if DEBUG
            StreamWriter errorsFile = File.CreateText("errors.csv");
#endif

            stopWatch.Restart();

            while (epoch_to_run < epochsCount && error > acceptableError)
            {
                epoch_to_run++;
                error = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    Run(inputs[i]);
                    BackPropagationMethod(outputs[i]);
                    error += Error(outputs[i]);
                }
                errorsFile.WriteLine(error);
                OnTrainProgress((epoch_to_run * 1.0) / epochsCount, error, stopWatch.Elapsed);
            }

#if DEBUG
            errorsFile.Close();
#endif

            OnTrainProgress(1.0, error, stopWatch.Elapsed);

            stopWatch.Stop();

            return error;
        }

        protected override double[] Compute(double[] input)
        {
            Run(input);
            return layerValues.Last().Take(layerValues.Last().Length - 1).ToArray();
        }
    }
}