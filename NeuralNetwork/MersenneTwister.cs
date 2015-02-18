/* 
   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

using System;
using System.Threading;

namespace NeuralNetwork
{
	/// <summary>MT19937 メルセンヌツイスタによる疑似乱数生成器を表します。</summary>
	public class MersenneTwister : Random
	{
		/// <summary>指定されたシード値を使用して、<see cref="MersenneTwister"/> クラスの新しいインスタンスを初期化します。</summary>
		/// <param name="seed">疑似乱数生成器を初期化するシード値を指定します。</param>
		public MersenneTwister(int seed)
		{
			_vector[0] = unchecked((uint)seed);
			for (_vectorIndex = 1; _vectorIndex < _vector.Length; _vectorIndex++)
				_vector[_vectorIndex] = (uint)(1812433253U * (_vector[_vectorIndex - 1] ^ (_vector[_vectorIndex - 1] >> 30)) + _vectorIndex);
		}

		// Period parameters
		const int M = 397;
		const uint UpperMask = 0x80000000U; // most significant w-r bits
		const uint LowerMask = 0x7fffffffU; // least significant r bits
		static readonly uint[] _mag01 = new[] { 0x0U, 0x9908b0dfU }; // mag01[x] = x * AVector  for x=0,1

		uint[] _vector = new uint[624];
		int _vectorIndex;

		/// <summary>0 以上で <see cref="UInt32.MaxValue"/> 以下の乱数を返します。</summary>
		/// <returns>0 以上 <see cref="UInt32.MaxValue"/> 以下の 32 ビット符号なし整数。</returns>
		public uint NextUInt32()
		{
			if (_vectorIndex >= _vector.Length)
			{
				// generate N words at one time
				for (int i = 0; i < _vector.Length; i++)
				{
					var temp = (_vector[i] & UpperMask) | (_vector[(i + 1) % _vector.Length] & LowerMask);
					_vector[i] = _vector[(i + M) % _vector.Length] ^ (temp >> 1) ^ _mag01[temp & 1];
				}
				_vectorIndex = 0;
			}
			var result = _vector[_vectorIndex++];
			// Tempering
			result ^= (result >> 11);
			result ^= (result << 7) & 0x9d2c5680U;
			result ^= (result << 15) & 0xefc60000U;
			result ^= (result >> 18);
			return result;
		}

		/// <summary>指定した範囲内の乱数を返します。</summary>
		/// <param name="minValue">返される乱数の包括的下限値を指定します。</param>
		/// <param name="maxValue">返される乱数の排他的上限値を指定します。 <paramref name="maxValue"/> は <paramref name="minValue"/> 以上である必要があります。</param>
		/// <returns>
		/// <paramref name="minValue"/> 以上で <paramref name="maxValue"/> 未満の 32 ビット符号付整数。
		/// つまり、戻り値の範囲に <paramref name="minValue"/> は含まれますが <paramref name="maxValue"/> は含まれません。
		/// <paramref name="minValue"/> が <paramref name="maxValue"/> と等しい場合は、<paramref name="minValue"/> が返されます。
		/// </returns>
		public override int Next(int minValue, int maxValue) { return (int)(NextUInt32() / ((double)uint.MaxValue + 1) * ((long)maxValue - minValue) + minValue); }

		/// <summary>指定した最大値より小さい 0 以上の乱数を返します。</summary>
		/// <param name="maxValue">生成される乱数の排他的上限値を指定します。 <paramref name="maxValue"/> は 0 以上である必要があります。</param>
		/// <returns>
		/// 0 以上で <paramref name="maxValue"/> 未満の 32 ビット符号付き整数。
		/// つまり、通常は戻り値の範囲に 0 は含まれますが、<paramref name="maxValue"/> は含まれません。
		/// ただし、<paramref name="maxValue"/> が 0 の場合は、<paramref name="maxValue"/> が返されます。
		/// </returns>
		public override int Next(int maxValue) { return Next(0, maxValue); }

		/// <summary>0 以上で <see cref="Int32.MaxValue"/> より小さい乱数を返します。</summary>
		/// <returns>0 以上で <see cref="Int32.MaxValue"/> より小さい 32 ビット符号付整数。</returns>
		public override int Next() { return Next(0, int.MaxValue); }

		/// <summary>0.0 以上 1.0 未満の乱数を返します。</summary>
		/// <returns>0.0 以上 1.0 未満の倍精度浮動小数点数。</returns>
		public override double NextDouble() { return NextUInt32() / ((double)uint.MaxValue + 1); }

		/// <summary>指定したバイト配列の要素に乱数を格納します。</summary>
		/// <param name="buffer">乱数を格納するバイト配列を指定します。。</param>
		public override void NextBytes(byte[] buffer)
		{
			uint sample = 0;
			for (int i = 0; i < buffer.Length; i++)
			{
				if (i % 4 == 0)
					sample = NextUInt32();
				buffer[i] = (byte)(sample & 0xff);
				sample >>= 8;
			}
		}

		/// <summary>0.0 以上 1.0 未満の乱数を返します。</summary>
		/// <returns>0.0 以上 1.0 未満の倍精度浮動小数点数。</returns>
		protected override double Sample() { return NextDouble(); }
	}
}
