using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	/// <summary>LINQ を拡張するメソッドを定義します。</summary>
	public static class LinqExtensions
	{
		/// <summary>シーケンスを指定された数で先頭からグループ化します。</summary>
		/// <typeparam name="T"><paramref name="source"/> の要素の型。</typeparam>
		/// <param name="source">グループ化するシーケンス。</param>
		/// <param name="count">1 グループに含める最大要素数。</param>
		/// <returns>入力シーケンスを先頭から指定された数の要素でグループ化した <see cref="IEnumerable&lt;T&gt;"/></returns>
		/// <exception cref="ArgumentNullException"><paramref name="source"/> は <c>null</c> です。</exception>
		/// <exception cref="ArgumentException"><paramref name="count"/> が 1 未満です。</exception>
		public static IEnumerable<IReadOnlyList<T>> Partition<T>(this IEnumerable<T> source, int count)
		{
			if (source == null)
				throw new ArgumentNullException("source");
			if (count <= 0)
				throw new ArgumentException("count は 1 以上である必要があります。", "count");
			return PartitionIterator(source, count);
		}

		static IEnumerable<IReadOnlyList<T>> PartitionIterator<T>(IEnumerable<T> source, int count)
		{
			List<T> result = new List<T>(count);
			foreach (var item in source)
			{
				result.Add(item);
				if (result.Count == count)
				{
					yield return result;
					result = new List<T>(count);
				}
			}
			if (result.Count > 0)
				yield return result;
		}
	}
}
