using Newtonsoft.Json;

namespace TinyGPT.Autoretriever.Demo
{
	static class Program
	{
		static void Main(string[] args)
		{
			Console.WriteLine("Loading dictionary...");
#pragma warning disable CS8604 // Possible null reference argument.
			IReadOnlyDictionary<string, string> dict = Util.FixDict(JsonConvert.DeserializeObject<Dictionary<string, string>>(File.ReadAllText(args[0])));
#pragma warning restore CS8604 // Possible null reference argument.
			while(true){
				Console.Write("Enter prompt: ");
				string? str = Console.ReadLine();
				if (str is null) return;
				Console.WriteLine(Util.ResolveWordMeanings(dict, str));
			}
		}
	}
}