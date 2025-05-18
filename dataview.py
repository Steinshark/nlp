import tkinter as tk
from tkinter import scrolledtext
import os

class WebpageViewerApp:
    def __init__(self, root, webpages):
        self.root = root
        self.webpages = webpages
        self.current_index = 0

        # UI Elements
        self.text_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30)
        self.text_display.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        
        self.stats_label = tk.Label(root, text="", justify=tk.LEFT, font=("Courier", 10))
        self.stats_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)

        self.char_entry = tk.Entry(root, width=10)
        self.char_entry.grid(row=1, column=2, padx=10, pady=5)
        
        self.char_button = tk.Button(root, text="Calculate Ratio", command=self.calculate_char_ratio)
        self.char_button.grid(row=2, column=2, padx=10, pady=5)
        
        self.next_button = tk.Button(root, text="Next Webpage", command=self.display_next_webpage)
        self.next_button.grid(row=2, column=1, padx=10, pady=5)

        # Display the first webpage on initialization
        if self.webpages:
            self.display_webpage(self.webpages[self.current_index])

    def calculate_stats(self, text):
        alphabet = "abcdefghijklmnopqrstuvwxyz.?!,"
        text_len = len(text)
        
        good_count = sum(text.lower().count(char) for char in alphabet)
        alphabet_ratio = good_count / text_len if text_len > 0 else 0
        
        punct_count = sum(text.count(p) for p in ".!?\",")
        punct_ratio = punct_count / text_len if text_len > 0 else 0
        
        num_words = len(text.split())
        avg_word_length = text_len / num_words if num_words > 0 else 0
        
        comma_count = text.count(",")
        comma_ratio = comma_count / num_words if num_words > 0 else 0

        return {
            "Alphabet Ratio": alphabet_ratio,
            "Punctuation Ratio": punct_ratio,
            "Avg Word Length": avg_word_length,
            "Comma Ratio": comma_ratio,
        }

    def calculate_char_ratio(self):
        if self.current_index < len(self.webpages):
            text = self.webpages[self.current_index]
            entry_char = self.char_entry.get()

            if len(entry_char) == 0:
                self.stats_label.config(text="Please enter a character or word.")
                return

            char_count = text.count(entry_char)
            text_len = len(text)
            normalized_ratio = (char_count / text_len) / len(entry_char) if text_len > 0 else 0
            self.stats_label.config(text=f"Ratio for '{entry_char}': {normalized_ratio:.5f}")

    def display_webpage(self, webpage):
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, webpage)
        
        stats = self.calculate_stats(webpage)
        stats_text = "\n".join([f"{key}: {value:.3f}" for key, value in stats.items()])
        self.stats_label.config(text=stats_text)

    def display_next_webpage(self):
        self.current_index += 1
        if self.current_index < len(self.webpages):
            self.display_webpage(self.webpages[self.current_index])
        else:
            self.text_display.delete(1.0, tk.END)
            self.stats_label.config(text="")
            self.text_display.insert(tk.END, "No more webpages to display.")

    def load_webpages_from_directory(self, directory):
        self.webpages = []
        try:
            for filename in os.listdir(directory):
                if filename.endswith(".txt"):
                    file_path = os.path.join(directory, filename)
                    with open(file_path, "r", encoding="utf-8") as file:
                        file_content = file.read()
                        pages = file_content.split("<|endoftext|>")
                        self.webpages.extend(pages)
                        if len(self.webpages) >= 1000:
                            self.webpages = self.webpages[:1000]
                            break
            if self.webpages:
                self.current_index = 0
                self.display_webpage(self.webpages[self.current_index])
            else:
                self.text_display.delete(1.0, tk.END)
                self.text_display.insert(tk.END, "No valid webpages found in the directory.")
        except Exception as e:
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(tk.END, f"Error loading webpages: {e}")

# Example usage with dummy webpages
def main():
    root = tk.Tk()
    root.title("Webpage Viewer")

    app = WebpageViewerApp(root, [])
    app.load_webpages_from_directory("C:/data/nlp/crawl")

    root.mainloop()

if __name__ == "__main__":
    main()
