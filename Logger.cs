// Copyright (c) 2016 robosoup
// www.robosoup.com

using System;
using System.IO;

namespace Model
{
    public class Logger : StreamWriter
    {
        public Logger(string path) : base(path) { }

        public override void Write(char value)
        {
            base.Write(value);
            Console.Write(value);
        }

        public override void Write(string value)
        {
            base.Write(value);
            Console.Write(value);
        }

        public override void WriteLine()
        {
            base.WriteLine();
            Console.WriteLine();
        }

        public override void WriteLine(string value)
        {
            base.WriteLine(value);
            Console.WriteLine(value);
        }
    }
}
