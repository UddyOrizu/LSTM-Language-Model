// Copyright (c) 2016 robosoup
// www.robosoup.com

using System;
using System.Linq;

namespace Model
{
    /// <summary>
    /// Recurrent neural network layer implementing backpropagation through time.
    /// </summary>
    public class RNN : Layer
    {
        // Dimensions.
        private int size_buffer;
        private int size_output;
        private int size_input;
        private int size_total;

        // State.
        private double[][] node_output;
        private double[][] vcx;

        // Parameters.
        private double[] b_node_output;
        private double[][] w_node_output;

        // Gradients.
        private double[] db_node_output;
        private double[][] dw_node_output;

        // Caches.
        private double[] cb_node_output;
        private double[][] cw_node_output;

        public override int Count()
        {
            return size_output + size_total * size_output;
        }

        public RNN(int size_input, int size_output, int size_buffer)
        {
            this.size_input = size_input;
            this.size_output = size_output;
            this.size_buffer = size_buffer;
            size_total = size_input + size_output;

            ResetState();
            ResetParameters();
            ResetGradients();
            ResetCaches();
        }

        public override double[][] Forward(double[][] buffer, bool reset)
        {
            if (reset) node_output[0] = new double[size_output];
            else node_output[0] = node_output[size_buffer - 1].ToArray();

            for (var t = 1; t < size_buffer; t++)
            {
                buffer[t].CopyTo(vcx[t], 0);
                node_output[t - 1].CopyTo(vcx[t], size_input);

                var row_vcx_state = vcx[t];

                node_output[t] = new double[size_output];
                for (var j = 0; j < size_output; j++)
                {
                    var sum = b_node_output[j];

                    var row = w_node_output[j];
                    for (var i = 0; i < size_total; i++)
                        sum += row_vcx_state[i] * row[i];

                    node_output[t][j] = Tanh(sum);
                }
            }

            return node_output;
        }

        public override double[][] Backward(double[][] grads, double alpha)
        {
            var grads_out = new double[size_buffer][];
            var dy_prev = new double[size_output];

            for (var t = size_buffer - 1; t > 0; t--)
            {
                grads_out[t] = new double[size_output];
                var row_grads_out = grads_out[t];

                var dy = dy_prev.ToArray();
                dy_prev = new double[size_output];

                var row_vcx = vcx[t];

                for (var j = 0; j < size_output; j++)
                {
                    dy[j] += Clip(grads[t][j]);
                    dy[j] = dTanh(node_output[t][j]) * dy[j];
                    db_node_output[j] += dy[j];

                    var row_w_node_output = w_node_output[j];
                    var row_dw_node_output = dw_node_output[j];

                    for (var i = 0; i < size_total; i++)
                    {
                        row_dw_node_output[i] += row_vcx[i] * dy[j];

                        if (i < size_input)
                            row_grads_out[i] += row_w_node_output[i] * dy[j];
                        else
                            dy_prev[i - size_input] += row_w_node_output[i] * dy[j];
                    }
                }
            }

            Update(alpha);
            ResetGradients();

            return grads_out;
        }

        protected override void ResetState()
        {
            node_output = new double[size_buffer][];
            vcx = new double[size_buffer][];

            for (var i = 0; i < size_buffer; i++)
            {
                node_output[i] = new double[size_output];
                vcx[i] = new double[size_total];
            }
        }

        protected override void ResetParameters()
        {
            b_node_output = new double[size_output];
            w_node_output = new double[size_output][];

            for (var j = 0; j < size_output; j++)
            {
                w_node_output[j] = new double[size_total];
                for (var i = 0; i < size_total; i++)
                    w_node_output[j][i] = RandomWeight();
            }
        }

        protected override void ResetGradients()
        {
            db_node_output = new double[size_output];
            dw_node_output = new double[size_output][];

            for (var i = 0; i < size_output; i++)
                dw_node_output[i] = new double[size_total];
        }

        protected override void ResetCaches()
        {
            cb_node_output = new double[size_output];
            cw_node_output = new double[size_output][];

            for (var j = 0; j < size_output; j++)
                cw_node_output[j] = new double[size_input];
        }

        protected override void Update(double alpha)
        {
            for (var j = 0; j < size_output; j++)
            {
                cb_node_output[j] = rmsDecay * cb_node_output[j] + (1 - rmsDecay) * Math.Pow(db_node_output[j], 2);
                b_node_output[j] -= Clip(db_node_output[j]) * alpha / Math.Sqrt(cb_node_output[j] + 1e-6);

                for (var i = 0; i < size_input; i++)
                {
                    cw_node_output[j][i] = rmsDecay * cw_node_output[j][i] + (1 - rmsDecay) * Math.Pow(cw_node_output[j][i], 2);
                    w_node_output[j][i] -= Clip(dw_node_output[j][i]) * alpha / Math.Sqrt(cw_node_output[j][i] + 1e-6);
                }
            }
        }
    }
}
