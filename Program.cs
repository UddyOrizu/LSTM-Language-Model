// Copyright (c) 2016 robosoup
// www.robosoup.com

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Generator
{
    public class Program
    {
        private Random rnd = new Random();
        private int size_vocab;
        private string text;
        private double loss;

        // Network layers.
        private Layer layer1;
        private Layer layer2;
        private Layer layer3;

        // Hyperparameters.
        private const double epochs = 50;
        private const int size_hidden = 100;
        private const int size_buffer = 24;
        private const int sample_length = 500;
        private const int sample_count = 3;
        private double learning_rate = 1e-3;

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

            var i = 0;
            foreach (var item in Decode) Encode.Add(item, i++);

            var decay = learning_rate / epochs;
            layer1 = new LSTM(size_vocab, size_hidden, size_buffer);
            layer2 = new LSTM(size_hidden, size_hidden, size_buffer);
            layer3 = new SoftMax(size_hidden, size_vocab, size_buffer);

            var param_count = size_vocab * size_hidden * 2 + size_hidden * size_hidden * 3 + size_hidden * 2 + size_vocab;
            Console.WriteLine("[{0:H:mm:ss}] Learning {1:#,###0} parameters...", DateTime.Now, param_count);
            Console.WriteLine();

            using (var logger = new Logger("log.txt"))
            {
                for (var epoch = 0; epoch < epochs; epoch++)
                {
                    learning_rate = learning_rate * 1 / (1 + decay * epoch);

                    var pos = 0;
                    while (pos + size_buffer < text.Length)
                    {
                        // Fill buffer.
                        var buffer = FillBuffer(pos, Encode);

                        // Forward propagate activation.
                        var reset = pos == 0;
                        var probs = layer3.Forward(layer2.Forward(layer1.Forward(buffer, reset), reset), reset);

                        // Advance buffer.                       
                        var vx = new double[size_vocab];
                        pos += size_buffer - 1;
                        vx[Encode[text[pos]]] = 1;
                        AdvanceBuffer(buffer, vx);

                        // Calculate loss.
                        var grads = Loss(probs, buffer);

                        // Backward propagate gradients.
                        layer1.Backward(layer2.Backward(layer3.Backward(grads, learning_rate), learning_rate), learning_rate);
                    }

                    // Write results to log.
                    logger.WriteLine();
                    logger.WriteLine("[{0:H:mm:ss}] epoch: {1}  loss: {2:0.000}", DateTime.Now, epoch, loss);

                    // Sample progress.
                    for (var g = 0; g < sample_count; g++)
                    {
                        Generate(logger, Decode, Encode);
                        logger.WriteLine(new String('-', 40));
                    }
                }
            }

            Console.WriteLine("[{0:H:mm:ss}] Finished!", DateTime.Now);
            Console.ReadLine();
        }

        /// <summary>
        /// Calculate cross entropy loss.
        /// </summary>
        private double[][] Loss(double[][] probs, double[][] targets)
        {
            var ls = 0.0;
            var grads = new double[size_buffer][];
            for (var t = 1; t < size_buffer; t++)
            {
                grads[t] = probs[t].ToArray();
                for (var i = 0; i < size_vocab; i++)
                {
                    ls += -Math.Log(probs[t][i]) * targets[t][i];
                    grads[t][i] -= targets[t][i];
                }
            }
            ls = ls / size_buffer;
            loss = loss * 0.99 + ls * 0.01;
            return grads;
        }

        /// <summary>
        /// Fill buffer with specified number of characters.
        /// </summary>
        private double[][] FillBuffer(int offset, Dictionary<char, int> Encode)
        {
            // First position is unused.
            var buffer = new double[size_buffer][];
            for (var pos = 1; pos < size_buffer; pos++)
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
            for (var b = 1; b < size_buffer - 1; b++)
                buffer[b] = buffer[b + 1];
            buffer[size_buffer - 1] = vx;
        }

        /// <summary>
        /// Generate sequence of text using trained network.
        /// </summary>
        private void Generate(Logger logger, char[] Decode, Dictionary<char, int> Encode)
        {
            var buffer = FillBuffer(0, Encode);
            logger.Write(text.Substring(0, size_buffer - 1));
            for (var pos = 0; pos < sample_length; pos++)
            {
                var reset = pos == 0;
                var vys = layer3.Forward(layer2.Forward(layer1.Forward(buffer, reset), reset), reset);
                var ix = WeightedChoice(vys[size_buffer - 1]);
                var vx = new double[size_vocab];
                vx[ix] = 1;
                AdvanceBuffer(buffer, vx);
                logger.Write(Decode[ix]);
            }
            logger.WriteLine("\r\n");
            logger.Flush();
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