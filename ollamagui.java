import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.prefs.Preferences;

public class LlamaGuiJavaEnhanced extends JFrame {

    // --- Configuration ---
    private static final String LLAMA_API_URL = "http://127.0.0.1:8080/completion";

    // --- GUI Components ---
    private JTextArea chatArea, systemPromptArea;
    private JTextField promptField, maxTokensField, seedField;
    private JButton sendButton, stopButton;
    private JSlider tempSlider, topPSlider, repeatPenaltySlider;
    private JComboBox<String> promptTemplateComboBox;
    private JLabel statusLabel, tpsLabel;

    // --- State & Threading ---
    private volatile boolean isGenerating = false;
    private volatile boolean stopRequested = false;
    private HttpClient httpClient;
    private Thread generationThread;

    // --- Settings Persistence ---
    private Preferences prefs;

    public LlamaGuiJavaEnhanced() {
        // --- Preferences ---
        prefs = Preferences.userNodeForPackage(LlamaGuiJavaEnhanced.class);

        // --- Frame Setup ---
        setTitle("Llama.cpp Java GUI (Enhanced)");
        loadSettings(); // Load window size/pos before setting up
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                saveSettings();
            }
        });
        
        // --- Menu Bar ---
        setJMenuBar(createMenuBar());

        // --- Main Layout ---
        JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        splitPane.setDividerLocation(prefs.getInt("dividerLocation", 350));

        // --- Left Panel (Controls) ---
        JPanel controlsPanel = createControlsPanel();
        JScrollPane controlsScrollPane = new JScrollPane(controlsPanel);
        controlsScrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
        splitPane.setLeftComponent(controlsScrollPane);

        // --- Right Panel (Chat) ---
        JPanel chatPanel = createChatPanel();
        splitPane.setRightComponent(chatPanel);

        add(splitPane);

        // --- Finalize ---
        httpClient = HttpClient.newHttpClient();
        loadSettings(); // Load component values after they are created
    }

    private JMenuBar createMenuBar() {
        JMenuBar menuBar = new JMenuBar();
        JMenu fileMenu = new JMenu("File");

        JMenuItem newChat = new JMenuItem("New Chat");
        newChat.addActionListener(e -> chatArea.setText(""));

        JMenuItem saveChat = new JMenuItem("Save Chat...");
        saveChat.addActionListener(e -> saveChatToFile());

        JMenuItem loadChat = new JMenuItem("Load Chat...");
        loadChat.addActionListener(e -> loadChatFromFile());
        
        JMenuItem exit = new JMenuItem("Exit");
        exit.addActionListener(e -> {
            saveSettings();
            System.exit(0);
        });

        fileMenu.add(newChat);
        fileMenu.add(new JSeparator());
        fileMenu.add(saveChat);
        fileMenu.add(loadChat);
        fileMenu.add(new JSeparator());
        fileMenu.add(exit);
        menuBar.add(fileMenu);
        return menuBar;
    }

    private JPanel createControlsPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // System Prompt
        panel.add(new JLabel("System Prompt:") {{ setAlignmentX(Component.LEFT_ALIGNMENT); }});
        systemPromptArea = new JTextArea(5, 20);
        systemPromptArea.setLineWrap(true);
        systemPromptArea.setWrapStyleWord(true);
        JScrollPane systemPromptScrollPane = new JScrollPane(systemPromptArea);
        systemPromptScrollPane.setAlignmentX(Component.LEFT_ALIGNMENT);
        panel.add(systemPromptScrollPane);
        panel.add(Box.createRigidArea(new Dimension(0, 15)));

        // Prompt Template
        panel.add(new JLabel("Prompt Template:") {{ setAlignmentX(Component.LEFT_ALIGNMENT); }});
        String[] templates = {"Default (no template)", "ChatML", "Llama2"};
        promptTemplateComboBox = new JComboBox<>(templates);
        promptTemplateComboBox.setAlignmentX(Component.LEFT_ALIGNMENT);
        panel.add(promptTemplateComboBox);
        panel.add(Box.createRigidArea(new Dimension(0, 15)));

        // Parameters
        JPanel paramsPanel = new JPanel();
        paramsPanel.setLayout(new BoxLayout(paramsPanel, BoxLayout.Y_AXIS));
        paramsPanel.setBorder(new TitledBorder("Parameters"));
        paramsPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
        
        tempSlider = createSliderControl(paramsPanel, "Temperature", 0, 200, 80);
        topPSlider = createSliderControl(paramsPanel, "Top-P", 0, 100, 95);
        repeatPenaltySlider = createSliderControl(paramsPanel, "Repeat Penalty", 100, 150, 110);
        maxTokensField = createTextControl(paramsPanel, "Max Tokens", "1024");
        seedField = createTextControl(paramsPanel, "Seed (-1 for random)", "-1");
        
        panel.add(paramsPanel);

        // Status Area
        panel.add(Box.createVerticalGlue()); // Pushes status to the bottom
        statusLabel = new JLabel("Ready");
        statusLabel.setForeground(Color.GRAY);
        statusLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        panel.add(statusLabel);

        tpsLabel = new JLabel("Tokens/sec: N/A");
        tpsLabel.setForeground(Color.GRAY);
        tpsLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        panel.add(tpsLabel);

        return panel;
    }

    private JSlider createSliderControl(JPanel parent, String label, int min, int max, int initial) {
        parent.add(new JLabel(label));
        JSlider slider = new JSlider(min, max, initial);
        parent.add(slider);
        parent.add(Box.createRigidArea(new Dimension(0, 5)));
        return slider;
    }

    private JTextField createTextControl(JPanel parent, String label, String initial) {
        JPanel fieldPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        fieldPanel.add(new JLabel(label + ": "));
        JTextField textField = new JTextField(initial, 10);
        fieldPanel.add(textField);
        parent.add(fieldPanel);
        return textField;
    }

    private JPanel createChatPanel() {
        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        chatArea = new JTextArea();
        chatArea.setEditable(false);
        chatArea.setLineWrap(true);
        chatArea.setWrapStyleWord(true);
        chatArea.setFont(new Font("Monospaced", Font.PLAIN, 14));
        JScrollPane scrollPane = new JScrollPane(chatArea);
        panel.add(scrollPane, BorderLayout.CENTER);

        JPanel inputPanel = new JPanel(new BorderLayout(10, 0));
        promptField = new JTextField();
        promptField.setFont(new Font("SansSerif", Font.PLAIN, 14));
        
        JPanel buttonPanel = new JPanel(new GridLayout(1, 2, 5, 0));
        sendButton = new JButton("Send");
        stopButton = new JButton("Stop");
        stopButton.setEnabled(false);
        buttonPanel.add(sendButton);
        buttonPanel.add(stopButton);

        inputPanel.add(promptField, BorderLayout.CENTER);
        inputPanel.add(buttonPanel, BorderLayout.EAST);
        panel.add(inputPanel, BorderLayout.SOUTH);

        // Action listeners
        sendButton.addActionListener(e -> sendMessage());
        promptField.addActionListener(e -> sendMessage());
        stopButton.addActionListener(e -> {
            stopRequested = true;
            if (generationThread != null) {
                generationThread.interrupt(); // Interrupts the blocking read
            }
        });

        return panel;
    }

    private void sendMessage() {
        if (isGenerating) return;

        String userPrompt = promptField.getText().trim();
        if (userPrompt.isEmpty()) return;

        try {
            // Validate integer fields before starting generation
            Integer.parseInt(maxTokensField.getText());
            Integer.parseInt(seedField.getText());
        } catch (NumberFormatException e) {
            JOptionPane.showMessageDialog(this, "Max Tokens and Seed must be valid integers.", "Input Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        setGeneratingState(true);
        updateChatArea("You: " + userPrompt + "\n\n");
        promptField.setText("");

        generationThread = new Thread(() -> fetchModelResponse(userPrompt));
        generationThread.start();
    }

    private String applyPromptTemplate(String userPrompt) {
        String template = (String) promptTemplateComboBox.getSelectedItem();
        String system = systemPromptArea.getText().trim();
        String history = chatArea.getText(); // Full history for context

        switch (template) {
            case "ChatML":
                StringBuilder chatmlPrompt = new StringBuilder();
                if (!system.isEmpty()) {
                    chatmlPrompt.append("<|im_start|>system\n").append(system).append("<|im_end|>\n");
                }
                // A simple way to inject history, could be more sophisticated
                if (history.contains("Assistant:")) {
                    // This is a rough approximation. A real implementation would parse and rebuild the conversation.
                     chatmlPrompt.append(history.replace("You:", "<|im_start|>user\n").replace("Assistant:", "<|im_end|>\n<|im_start|>assistant\n"));
                }
                chatmlPrompt.append("<|im_start|>user\n").append(userPrompt).append("<|im_end|>\n<|im_start|>assistant\n");
                return chatmlPrompt.toString();
            case "Llama2":
                String B_INST = "[INST]", E_INST = "[/INST]";
                String B_SYS = "<<SYS>>\n", E_SYS = "\n<</SYS>>\n\n";
                if (system.isEmpty()) {
                    return B_INST + userPrompt + E_INST;
                } else {
                    return B_INST + B_SYS + system + E_SYS + userPrompt + E_INST;
                }
            default: // No template
                return userPrompt;
        }
    }

    private void fetchModelResponse(String userPrompt) {
        stopRequested = false;
        String fullPrompt = applyPromptTemplate(userPrompt);

        try {
            String jsonPayload = String.format(
                "{\"prompt\": \"%s\", \"stream\": true, \"temperature\": %.2f, \"top_p\": %.2f, \"repeat_penalty\": %.2f, \"n_predict\": %d, \"seed\": %d}",
                escapeJson(fullPrompt),
                tempSlider.getValue() / 100.0,
                topPSlider.getValue() / 100.0,
                repeatPenaltySlider.getValue() / 100.0,
                Integer.parseInt(maxTokensField.getText()),
                Integer.parseInt(seedField.getText())
            );

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(LLAMA_API_URL))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                    .build();
            
            SwingUtilities.invokeLater(() -> updateChatArea("Assistant: "));

            HttpResponse<InputStream> response = httpClient.send(request, HttpResponse.BodyHandlers.ofInputStream());

            if (response.statusCode() != 200) {
                throw new IOException("HTTP error code: " + response.statusCode());
            }

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body()))) {
                String line;
                while ((line = reader.readLine()) != null && !stopRequested) {
                    if (line.startsWith("data: ")) {
                        String jsonData = line.substring(6);
                        // Update UI with token
                        String token = parseJsonValue(jsonData, "content");
                        if (token != null) {
                            SwingUtilities.invokeLater(() -> updateChatArea(unescapeJson(token)));
                        }
                        // Check for stop condition and timings
                        if ("true".equals(parseJsonValue(jsonData, "stop"))) {
                            String tps = parseJsonValue(parseJsonValue(jsonData, "timings"), "predicted_per_second");
                            if (tps != null) {
                                final String finalTps = tps;
                                SwingUtilities.invokeLater(() -> tpsLabel.setText(String.format("Tokens/sec: %.2f", Double.parseDouble(finalTps))));
                            }
                            break;
                        }
                    }
                }
            }
        } catch (ConnectException e) {
             JOptionPane.showMessageDialog(this, "Connection failed. Is the llama.cpp server running?", "Connection Error", JOptionPane.ERROR_MESSAGE);
        } catch (IOException | InterruptedException e) {
            if (stopRequested) {
                SwingUtilities.invokeLater(() -> updateChatArea("\n[Generation stopped by user]"));
            } else {
                SwingUtilities.invokeLater(() -> updateChatArea("\n[Error: " + e.getMessage() + "]"));
                e.printStackTrace();
            }
        } finally {
            SwingUtilities.invokeLater(() -> setGeneratingState(false));
            SwingUtilities.invokeLater(() -> updateChatArea("\n\n"));
        }
    }

    // --- Utility Methods ---

    private void setGeneratingState(boolean generating) {
        this.isGenerating = generating;
        SwingUtilities.invokeLater(() -> {
            sendButton.setEnabled(!generating);
            stopButton.setEnabled(generating);
            promptField.setEnabled(!generating);
            statusLabel.setText(generating ? "Generating..." : "Ready");
            if (!generating) {
                generationThread = null;
            } else {
                tpsLabel.setText("Tokens/sec: N/A");
            }
        });
    }

    private void updateChatArea(String text) {
        chatArea.append(text);
        chatArea.setCaretPosition(chatArea.getDocument().getLength());
    }

    private String parseJsonValue(String json, String key) {
        if (json == null) return null;
        String keyPattern = "\"" + key + "\":";
        int keyIndex = json.indexOf(keyPattern);
        if (keyIndex == -1) return null;

        int valueStartIndex = keyIndex + keyPattern.length();
        char firstChar = json.charAt(valueStartIndex);

        if (firstChar == '\"') { // It's a string
            int endIndex = json.indexOf('\"', valueStartIndex + 1);
            if (endIndex != -1) {
                return json.substring(valueStartIndex + 1, endIndex);
            }
        } else if (firstChar == '{') { // It's an object
            // Find matching closing brace (simple implementation)
            int braceCount = 1;
            int endIndex = valueStartIndex + 1;
            while(endIndex < json.length() && braceCount > 0) {
                if (json.charAt(endIndex) == '{') braceCount++;
                if (json.charAt(endIndex) == '}') braceCount--;
                endIndex++;
            }
            return json.substring(valueStartIndex, endIndex);
        }
        else { // It's a number or boolean
            int endIndex = valueStartIndex;
            while (endIndex < json.length() && ",}".indexOf(json.charAt(endIndex)) == -1) {
                endIndex++;
            }
            return json.substring(valueStartIndex, endIndex).trim();
        }
        return null;
    }

    private String escapeJson(String text) {
        return text.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }

    private String unescapeJson(String text) {
        return text.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\");
    }

    // --- File and Settings I/O ---

    private void saveChatToFile() {
        JFileChooser fileChooser = new JFileChooser();
        if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
                writer.write(chatArea.getText());
                JOptionPane.showMessageDialog(this, "Chat saved successfully!");
            } catch (IOException e) {
                JOptionPane.showMessageDialog(this, "Error saving file: " + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private void loadChatFromFile() {
        JFileChooser fileChooser = new JFileChooser();
        if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                chatArea.read(reader, null);
            } catch (IOException e) {
                JOptionPane.showMessageDialog(this, "Error loading file: " + e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private void saveSettings() {
        prefs.putInt("x", getX());
        prefs.putInt("y", getY());
        prefs.putInt("width", getWidth());
        prefs.putInt("height", getHeight());
        prefs.putInt("dividerLocation", ((JSplitPane) getContentPane().getComponent(0)).getDividerLocation());
        
        prefs.putInt("temp", tempSlider.getValue());
        prefs.putInt("topP", topPSlider.getValue());
        prefs.putInt("repeatPenalty", repeatPenaltySlider.getValue());
        prefs.put("maxTokens", maxTokensField.getText());
        prefs.put("seed", seedField.getText());
        prefs.put("systemPrompt", systemPromptArea.getText());
        prefs.putInt("promptTemplate", promptTemplateComboBox.getSelectedIndex());
    }

    private void loadSettings() {
        setBounds(prefs.getInt("x", 100), prefs.getInt("y", 100), 
                  prefs.getInt("width", 1100), prefs.getInt("height", 800));
        
        if (tempSlider != null) { // Check if components are initialized
            tempSlider.setValue(prefs.getInt("temp", 80));
            topPSlider.setValue(prefs.getInt("topP", 95));
            repeatPenaltySlider.setValue(prefs.getInt("repeatPenalty", 110));
            maxTokensField.setText(prefs.get("maxTokens", "1024"));
            seedField.setText(prefs.get("seed", "-1"));
            systemPromptArea.setText(prefs.get("systemPrompt", ""));
            promptTemplateComboBox.setSelectedIndex(prefs.getInt("promptTemplate", 0));
        }
    }

    // --- Main Method ---
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception e) {
                e.printStackTrace();
            }
            new LlamaGuiJavaEnhanced().setVisible(true);
        });
    }
}
