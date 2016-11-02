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

        // Hyperparameters.
        private const int size_hidden = 150;
        private const int size_buffer = 15;
        private const int sample_length = 160;
        private const double epochs = 100;
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

            layer1 = new LSTM(size_vocab, size_hidden, size_buffer);
            layer2 = new SoftMax(size_hidden, size_vocab, size_buffer);

            Console.WriteLine("[{0:H:mm:ss}] Starting...", DateTime.Now);
            Console.WriteLine();

            var decay = learning_rate / epochs;

            using (var logger = new Logger("log.txt"))
            {
                for (var epoch = 0; epoch < epochs; epoch++)
                {
                    learning_rate = learning_rate * 1 / (1 + decay * epoch);
                    var buffer = PrimeBuffer(Encode);

                    for (var pos = size_buffer; pos < text.Length; pos++)
                    {
                        // Forward propagate.
                        var reset = pos == size_buffer;
                        var vys = layer2.Forward(layer1.Forward(buffer, reset), reset);

                        // Advance buffer.
                        var vx = new double[size_vocab];
                        vx[Encode[text[pos]]] = 1;
                        AdvanceBuffer(buffer, vx);

                        // Calculate loss.
                        var grads = Loss(vys, buffer);

                        // Backward propagate.
                        layer1.Backward(layer2.Backward(grads, learning_rate), learning_rate);

                        // Intra-epoch update.
                        if (pos % 100 == 0) Console.WriteLine("epoch: {0}  pos: {1}  loss: {2:0.000}", epoch, pos, loss);
                    }

                    // Write results to log.
                    logger.WriteLine();
                    logger.WriteLine("[{0:H:mm:ss}] epoch: {1}  loss: {2:0.000}", DateTime.Now, epoch, loss);

                    // Sample progress.
                    for (var g = 0; g < 5; g++)
                        Generate(logger, Decode, Encode, buffer);
                }
            }

            Console.WriteLine("[{0:H:mm:ss}] Finished!", DateTime.Now);
            Console.ReadLine();
        }

        /// <summary>
        /// Calculate cross entropy loss.
        /// </summary>
        private double[][] Loss(double[][] vys, double[][] targets)
        {
            var ls = 0.0;
            var grads = new double[size_buffer][];
            for (var t = 1; t < size_buffer; t++)
            {
                grads[t] = vys[t].ToArray();
                for (var i = 0; i < size_vocab; i++)
                {
                    ls += -Math.Log(vys[t][i]) * targets[t][i];
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
        private double[][] PrimeBuffer(Dictionary<char, int> Encode)
        {
            var buffer = new double[size_buffer][];
            for (var pos = 0; pos < size_buffer; pos++)
            {
                buffer[pos] = new double[size_vocab];
                buffer[pos][Encode[text[pos]]] = 1;
            }
            return buffer;
        }

        /// <summary>
        /// Read next character into buffer.
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="vx"></param>
        private static void AdvanceBuffer(double[][] buffer, double[] vx)
        {
            for (var b = 0; b < size_buffer - 1; b++)
                buffer[b] = buffer[b + 1];
            buffer[size_buffer - 1] = vx;
        }

        /// <summary>
        /// Generate sequence of text using trained network.
        /// </summary>
        private void Generate(Logger logger, char[] Decode, Dictionary<char, int> Encode, double[][] buffer)
        {
            buffer = PrimeBuffer(Encode);
            logger.Write(text.Substring(0, size_buffer));
            for (var pos = size_buffer; pos < sample_length; pos++)
            {
                var reset = pos == size_buffer;
                var vys = layer2.Forward(layer1.Forward(buffer, reset), reset);
                var ix = WeightedChoice(vys[size_buffer - 1]);
                var vx = new double[size_vocab];
                vx[ix] = 1;
                AdvanceBuffer(buffer, vx);
                logger.Write(Decode[ix]);
            }
            logger.WriteLine("\r\n");
        }

        /// <summary>
        ///  Select next character from weighted random distribution.
        /// </summary>
        /// <param name="vy"></param>
        /// <returns></returns>
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