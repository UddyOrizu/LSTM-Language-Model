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
        private double loss_p;

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
        private double learning_rate = 5e-3;

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

            layer1 = new LSTM(size_vocab, size_hidden, size_buffer);
            layer2 = new LSTM(size_hidden, size_hidden, size_buffer);
            layer3 = new SoftMax(size_hidden, size_vocab, size_buffer);

            using (var logger = new Logger("log.txt"))
            {
                var param_count = size_vocab * size_hidden * 2 + size_hidden * size_hidden * 3 + size_hidden * 2 + size_vocab;
                logger.WriteLine("[{0:H:mm:ss}] Learning {1:#,###0} parameters...", DateTime.Now, param_count);
                logger.WriteLine();

                for (var epoch = 0; epoch < epochs; epoch++)
                {
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

                    // Sample progress.
                    logger.WriteLine("[{0:H:mm:ss}] epoch: {1}  learning rate: {2:0.0000}  loss: {3:0.000}", DateTime.Now, epoch, learning_rate, loss);
                    for (var g = 0; g < sample_count; g++)
                    {
                        logger.WriteLine(new String('-', 40));
                        Generate(logger, Decode, Encode);
                    }
                    logger.WriteLine(new String('-', 40));
                    logger.WriteLine();
                    logger.Flush();

                    // Adjust learning rate.
                    if (loss_p - loss > 0) learning_rate += learning_rate * 0.01;
                    else learning_rate -= learning_rate * 0.02;
                    loss_p = loss_p * 0.8 + loss * 0.2;
                }

                logger.WriteLine("[{0:H:mm:ss}] Finished!", DateTime.Now);
            }
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
            ls = ls / (size_buffer - 1);
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