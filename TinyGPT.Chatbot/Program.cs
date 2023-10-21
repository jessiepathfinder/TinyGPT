using Newtonsoft.Json;
using System.Runtime.Serialization.Formatters;
using System.Transactions;
using TinyGPT.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.Chatbot
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string datadir = args[0];
			string model = args[1];
			if (!datadir.EndsWith(Path.DirectorySeparatorChar))
			{
				datadir += Path.DirectorySeparatorChar;
			}
			Console.WriteLine("Loading dictionary...");
			IReadOnlyDictionary<string, ushort>? dict = JsonConvert.DeserializeObject<IReadOnlyDictionary<string, ushort>>(File.ReadAllText(datadir + "encoder.json"));
			if (dict is null)
			{
				Console.WriteLine("Null encoder dictionary");
				return;
			}
			int maxlen = 0;
			int tokenclasses = 0;
			foreach (KeyValuePair<string, ushort> keyValuePair in dict)
			{
				maxlen = Math.Max(maxlen, keyValuePair.Key.Length);
				int val = keyValuePair.Value;
				tokenclasses = Math.Max(val, tokenclasses);
			}

			//2 magic token types
			tokenclasses += 3;
			string[] decode = new string[tokenclasses + 1];
			int maxtokensize = 0;
			foreach(KeyValuePair<string, ushort> keyValuePair in dict){
				string key = keyValuePair.Key;
				decode[keyValuePair.Value + 2] = key;
				maxtokensize = Math.Max(maxtokensize, key.Length);
			}
			Console.WriteLine("Loading model...");
			bool usecuda = cuda_is_available();
			if( usecuda ) {
				InitializeDeviceType(DeviceType.CUDA);
			} else{
				Console.WriteLine("WARNING: Your computer does not have a NVIDIA CUDA Graphics Card, or lacks the needed graphics drivers (I will be very slow)");
			}
			FullGPTDecoderUnit themodel;
			int maxcontext;
			switch(model){
				case "nano-v1":
					{
						const int latentTokenSize = 512;
						maxcontext = 2048;
						const int attentionHeads = 8;
						const int firstTierAttentionDepth = 8;

						ModuleList<BERTDictionaryItem> dictionaryItems = new ModuleList<BERTDictionaryItem>();
						for (int i = 0; i < tokenclasses; ++i)
						{
							dictionaryItems.Add(new BERTDictionaryItem("", latentTokenSize));
						}
						GPTDecoderUnitV1 notchatgpt = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, tokenclasses, firstTierAttentionDepth, 1e-8);
						themodel = new SimpleFullGPTDecoderUnit(dictionaryItems, notchatgpt, "");

					}
					break;
				default:
					Console.WriteLine("unknown model!");
					return;
			}
			themodel.to(ScalarType.BFloat16);
			themodel.load(datadir + model + ".model");
			themodel.eval();
			if (usecuda)
			{
				themodel.to(CUDA);
			}

			Span<ushort> buffer = stackalloc ushort[maxcontext];
			int maxinsize = maxcontext - 2;
			Console.WriteLine("Ready!");
			while(true){
				Console.Write("User: ");
				string? input = Console.ReadLine();
				if(input is null){
					continue;
				}
				Console.Write("TinyGPT: ");
				int intokens = Transformer.Tokenize(dict, buffer, input, maxtokensize, 2);
				if(intokens > maxinsize){
					Console.WriteLine("too big!");
					continue;
				}
				buffer[intokens] = 0; //[STARTGPT]
				int prev = 1;
				for (int i = intokens + 1; i < maxcontext; ++i){
					double best = double.NegativeInfinity;
					int bestindex = 1;
					Tensor tensor;
					using (var ds = NewDisposeScope()){
						tensor = themodel.Forward(buffer.Slice(0, i)).softmax(0).cpu();
						tensor.MoveToOuterDisposeScope();
					}
					using(tensor){
						for (int z = 0; z < tokenclasses; ++z)
						{
							double my = tensor[z].ToScalar().ToDouble();
							if (my > best)
							{
								if(z == prev){
									//continue;
								}
								best = my;
								bestindex = z;
							}
						}
					}
					
					if (bestindex == 1){
						break;
					}
					buffer[i] = (ushort)bestindex;
					prev = bestindex;
					string? str = decode[bestindex];

					if (str is null){
						str = " invalid_word_" + bestindex;
					}
					Console.Write(str);
				}
				Console.WriteLine();
				Console.WriteLine("==================================================");
				Console.WriteLine();

			}
		}
	}
}