﻿// Copyright (c) 2016 robosoup
// www.robosoup.com

using System;

namespace Generator
{
    /// <summary>
    /// Base class for all neural network layers.
    /// </summary>
    public abstract class Layer
    {
        protected const double rmsDecay = 0.95;

        protected readonly Random random = new Random();

        public abstract double[][] Forward(double[][] buffer, bool reset);

        public abstract double[][] Backward(double[][] grads, double alpha);

        protected abstract void ResetState();

        protected abstract void ResetParameters();

        protected abstract void ResetGradients();

        protected abstract void ResetCaches();

        protected abstract void Update(double alpha);

        /// <summary>
        /// Prevent gradient explosions.
        /// </summary>
        protected static double Clip(double x)
        {
            if (x < -1.0) return -1.0;
            if (x > 1.0) return 1.0;
            return x;
        }

        /// <summary>
        /// Squashing function returning values between zero and plus one.
        /// </summary>
        protected static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        /// <summary>
        /// Squashing function returning values between minus one and plus one.
        /// </summary>
        protected static double Tanh(double x)
        {
            return Math.Tanh(x);
        }

        /// <summary>
        /// Derivative of sigmoid function.
        /// </summary>
        protected static double dSigmoid(double x)
        {
            return (1 - x) * x;
        }

        /// <summary>
        /// Derivative of hyperbolic tangent function.
        /// </summary>
        protected static double dTanh(double x)
        {
            return 1 - x * x;
        }
    }
}