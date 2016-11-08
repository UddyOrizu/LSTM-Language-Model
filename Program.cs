// Copyright (c) 2016 robosoup
// www.robosoup.com

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Model
{
    public class Program
    {
        private Random rnd = new Random();
        private string text;
        private double loss;
        private double loss_p;
        private int size_vocab;

        // Network layers.
        private const int size_hidden = 256;
        private Layer layer1;
        private Layer layer2;
        private Layer layer3;

        static void Main()
        {
            new Program();
        }

        public Program()
        {
            using (var sr = new StreamReader("aesop.txt")) text = sr.ReadToEnd();
            var Decode = text.Distinct().OrderBy(x => x).ToArray();
            var Encode = new Dictionary<char, int>();
            size_vocab = Decode.Length;
            loss_p = Math.Log(size_vocab);

            var i = 0;
            foreach (var item in Decode) Encode.Add(item, i++);

            Layer.BufferSize = 24;
            Layer.LearningRate = 1e-3;
            layer1 = new LSTM(size_vocab, size_hidden);
            layer2 = new LSTM(size_hidden, size_hidden);
            layer3 = new SoftMax(size_hidden, size_vocab);

            var param_count = layer1.Count() + layer2.Count() + layer3.Count();

            using (var logger = new Logger("log.txt"))
            {
                logger.WriteLine("[{0:H:mm:ss}] Learning {1:#,###0} parameters...", DateTime.Now, param_count);
                logger.WriteLine();

                var iter = 0;
                while (true)
                {
                    var pos = 0;
                    while (pos + Layer.BufferSize < text.Length)
                    {
                        // Fill buffer.
                        var buffer = FillBuffer(pos, Encode);

                        // Forward propagate activation.
                        var reset = pos == 0;
                        var probs = layer3.Forward(layer2.Forward(layer1.Forward(buffer, reset), reset), reset);

                        // Advance buffer.                       
                        var vx = new double[size_vocab];
                        pos += Layer.BufferSize - 1;
                        vx[Encode[text[pos]]] = 1;
                        AdvanceBuffer(buffer, vx);

                        // Calculate loss.
                        var grads = Loss(probs, buffer);

                        // Backward propagate gradients.
                        layer1.Backward(layer2.Backward(layer3.Backward(grads)));
                    }

                    // Sample progress.
                    logger.WriteLine("[{0:H:mm:ss}] iteration: {1}  learning rate: {2:0.0000}  loss: {3:0.000}", DateTime.Now, iter, Layer.LearningRate, loss);
                    if (iter % 5 == 0)
                    {
                        logger.WriteLine(new String('-', 60));
                        Generate(logger, Decode, Encode);
                        logger.WriteLine(new String('-', 60));
                        logger.WriteLine();
                        logger.Flush();
                    }

                    // Adjust learning rate.
                    if (loss_p - loss > 0) Layer.LearningRate *= 1.01;
                    else Layer.LearningRate *= 0.98;
                    loss_p = loss_p * 0.8 + loss * 0.2;

                    iter++;
                }
            }
        }

        /// <summary>
        /// Calculate cross entropy loss.
        /// </summary>
        private double[][] Loss(double[][] probs, double[][] targets)
        {
            var ls = 0.0;
            var grads = new double[Layer.BufferSize][];
            for (var t = 1; t < Layer.BufferSize; t++)
            {
                grads[t] = probs[t].ToArray();
                for (var i = 0; i < size_vocab; i++)
                {
                    ls += -Math.Log(probs[t][i]) * targets[t][i];
                    grads[t][i] -= targets[t][i];
                }
            }
            ls = ls / (Layer.BufferSize - 1);
            loss = loss * 0.99 + ls * 0.01;
            return grads;
        }

        /// <summary>
        /// Fill buffer with specified number of characters.
        /// </summary>
        private double[][] FillBuffer(int offset, Dictionary<char, int> Encode)
        {
            // First position is unused.
            var buffer = new double[Layer.BufferSize][];
            for (var pos = 1; pos < Layer.BufferSize; pos++)
            {
                buffer[pos] = new double[size_vocab];
                buffer[pos][Encode[text[pos + offset - 1]]] = 1;
            }
            return buffer;
        }

        /// <summary>
        /// Read next character into buffer.
        /// </summary>
        private static void AdvanceBuffer(double[][] buffer, double[] vx)
        {
            for (var b = 1; b < Layer.BufferSize - 1; b++)
                buffer[b] = buffer[b + 1];
            buffer[Layer.BufferSize - 1] = vx;
        }

        /// <summary>
        /// Generate sequence of text using trained network.
        /// </summary>
        private void Generate(Logger logger, char[] Decode, Dictionary<char, int> Encode)
        {
            var buffer = FillBuffer(0, Encode);
            for (var pos = 0; pos < 500; pos++)
            {
                var reset = pos == 0;
                var probs = layer3.Forward(layer2.Forward(layer1.Forward(buffer, reset), reset), reset);
                var ix = WeightedChoice(probs[Layer.BufferSize - 1]);
                var vx = new double[size_vocab];
                vx[ix] = 1;
                AdvanceBuffer(buffer, vx);
                logger.Write(Decode[ix]);
            }
            logger.WriteLine();
        }

        /// <summary>
        ///  Select next character from weighted random distribution.
        /// </summary>
        private int WeightedChoice(double[] vy)
        {
            var val = rnd.NextDouble();
            for (var i = 0; i < vy.Length; i++)
            {
                if (val <= vy[i]) return i;
                val -= vy[i];
            }
            throw new Exception("Not in dictionary!");
        }
    }
}