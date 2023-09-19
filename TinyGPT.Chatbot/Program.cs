using Newtonsoft.Json;
using System.Runtime.Serialization.Formatters;
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
			Console.WriteLine("Loading dictioanry...");
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
				tokenclasses = Math.Max(keyValuePair.Value, tokenclasses);
			}

			//3 magic token types
			tokenclasses += 3;
			string[] decode = new string[tokenclasses + 3];
			int maxtokensize = 0;
			foreach(KeyValuePair<string, ushort> keyValuePair in dict){
				string key = keyValuePair.Key;
				decode[keyValuePair.Value + 3] = key;
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
						ModuleList<BERTDictionaryItem> dictionaryItems = new ModuleList<BERTDictionaryItem>();
						for (int i = 0; i < tokenclasses; ++i)
						{
							dictionaryItems.Add(new BERTDictionaryItem("", 256));
						}

						themodel = new FullGPTDecoderUnitV1("TinyGPT", dictionaryItems, new GPTDecoderV1(16, 2, 256, tokenclasses, 128, 1024, 1024, ""));
						maxcontext = 1024;
					}
					break;
				default:
					Console.WriteLine("unknown model!");
					return;
			}
			themodel.load(datadir + model + ".model");
			themodel.eval();
			if (usecuda)
			{
				themodel.to(CUDA, ScalarType.Float32);
			}

			Span<ushort> buffer = stackalloc ushort[maxcontext];
			int maxinsize = maxcontext - 2;
			Console.WriteLine("Ready!");
			while(true){
				Console.Write("User: ");
				string input = Console.ReadLine();
				Console.Write("TinyGPT: ");
				int intokens = Transformer.Tokenize(dict, buffer, input, maxtokensize);
				if(intokens > maxinsize){
					Console.WriteLine("too big!");
					continue;
				}
				buffer[intokens] = 0; //[STARTGPT]
				for(int i = intokens + 2; i < maxcontext; ++i){
					buffer[i] = 2; //[MASK]
					float best = -1;
					int bestindex = 1;
					Tensor tensor = themodel.forward(buffer.Slice(0, i + 1)).cpu();
					for(int z = 0; z < tokenclasses; ++z){
						float my = tensor[z].ToScalar().ToSingle();
						if(my > best){
							best = my;
							bestindex = z;
						}
					}
					if(bestindex == 1){
						break;
					}
					string? str = decode[bestindex];
					buffer[i] = (ushort)bestindex;
					if (str is null){
						continue;
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