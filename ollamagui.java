import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;

public class LlamaGuiJava extends JFrame {

    // --- Configuration ---
    private static final String LLAMA_API_URL = "http://127.0.0.1:8080/completion";

    // --- GUI Components ---
    private JTextArea chatArea;
    private JTextField promptField;
    private JButton sendButton;
    private JSlider tempSlider, topPSlider, repeatPenaltySlider;
    private JTextField maxTokensField;
    private JLabel statusLabel;

    private volatile boolean isGenerating = false;

    public LlamaGuiJava() {
        // --- Frame Setup ---
        setTitle("Llama.cpp Java GUI");
        setSize(1000, 700);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        // --- Main Layout ---
        JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        splitPane.setDividerLocation(280);

        // --- Left Panel (Controls) ---
        JPanel controlsPanel = createControlsPanel();
        splitPane.setLeftComponent(controlsPanel);

        // --- Right Panel (Chat) ---
        JPanel chatPanel = createChatPanel();
        splitPane.setRightComponent(chatPanel);

        add(splitPane);
    }

    private JPanel createControlsPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        panel.add(new JLabel("Model Parameters") {
            {
                setFont(getFont().deriveFont(Font.BOLD, 16f));
                setAlignmentX(Component.LEFT_ALIGNMENT);
            }
        });
        panel.add(Box.createRigidArea(new Dimension(0, 20)));

        // Sliders are integer-based, so we'll use a multiplier
        tempSlider = createSliderControl(panel, "Temperature", 0, 200, 80); // 0.0 to 2.0
        topPSlider = createSliderControl(panel, "Top-P", 0, 100, 95); // 0.0 to 1.0
        repeatPenaltySlider = createSliderControl(panel, "Repeat Penalty", 100, 150, 110); // 1.0 to 1.5

        panel.add(Box.createRigidArea(new Dimension(0, 10)));
        maxTokensField = createTextControl(panel, "Max Tokens", "1024");

        panel.add(Box.createVerticalGlue()); // Pushes status label to the bottom

        statusLabel = new JLabel("Ready");
        statusLabel.setForeground(Color.GRAY);
        panel.add(statusLabel);

        return panel;
    }

    private JSlider createSliderControl(JPanel parent, String label, int min, int max, int initial) {
        parent.add(new JLabel(label) {{ setAlignmentX(Component.LEFT_ALIGNMENT); }});
        JSlider slider = new JSlider(min, max, initial);
        slider.setAlignmentX(Component.LEFT_ALIGNMENT);
        parent.add(slider);
        parent.add(Box.createRigidArea(new Dimension(0, 15)));
        return slider;
    }

    private JTextField createTextControl(JPanel parent, String label, String initial) {
        JPanel fieldPanel = new JPanel(new BorderLayout());
        fieldPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
        fieldPanel.add(new JLabel(label), BorderLayout.WEST);
        JTextField textField = new JTextField(initial);
        fieldPanel.add(textField, BorderLayout.CENTER);
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
        sendButton = new JButton("Send");

        inputPanel.add(promptField, BorderLayout.CENTER);
        inputPanel.add(sendButton, BorderLayout.EAST);
        panel.add(inputPanel, BorderLayout.SOUTH);

        // Add action listeners
        sendButton.addActionListener(e -> sendMessage());
        promptField.addActionListener(e -> sendMessage());

        return panel;
    }

    private void sendMessage() {
        if (isGenerating) return;

        String prompt = promptField.getText().trim();
        if (prompt.isEmpty()) return;

        setGeneratingState(true);
        updateChatArea("You: " + prompt + "\n\n");
        promptField.setText("");

        // Run network request in a background thread to not freeze the GUI
        new Thread(() -> fetchModelResponse(prompt)).start();
    }
    
    // Manual JSON escaping for the prompt
    private String escapeJson(String text) {
        return text.replace("\\", "\\\\")
                   .replace("\"", "\\\"")
                   .replace("\b", "\\b")
                   .replace("\f", "\\f")
                   .replace("\n", "\\n")
                   .replace("\r", "\\r")
                   .replace("\t", "\\t");
    }

    private void fetchModelResponse(String prompt) {
        try {
            // Build JSON payload manually
            String escapedPrompt = escapeJson(prompt);
            String jsonPayload = String.format(
                "{\"prompt\": \"%s\", \"stream\": true, \"temperature\": %.2f, \"top_p\": %.2f, \"repeat_penalty\": %.2f, \"n_predict\": %d}",
                escapedPrompt,
                tempSlider.getValue() / 100.0,
                topPSlider.getValue() / 100.0,
                repeatPenaltySlider.getValue() / 100.0,
                Integer.parseInt(maxTokensField.getText())
            );

            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(LLAMA_API_URL))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                    .build();
            
            SwingUtilities.invokeLater(() -> updateChatArea("Assistant: "));

            // Use an InputStream to handle the streaming response
            HttpResponse<java.io.InputStream> response = client.send(request, HttpResponse.BodyHandlers.ofInputStream());

            if (response.statusCode() != 200) {
                throw new RuntimeException("Failed : HTTP error code : " + response.statusCode());
            }

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.body()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("data: ")) {
                        String jsonData = line.substring(6);
                        // Manual, simple JSON parsing to find the "content"
                        int contentIndex = jsonData.indexOf("\"content\":\"");
                        if (contentIndex != -1) {
                            int startIndex = contentIndex + 11;
                            int endIndex = jsonData.indexOf("\"", startIndex);
                            if (endIndex != -1) {
                                String token = jsonData.substring(startIndex, endIndex);
                                // Unescape common sequences from JSON string
                                String unescapedToken = token.replace("\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\");
                                SwingUtilities.invokeLater(() -> updateChatArea(unescapedToken));
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            final String errorMessage = "\n\n[Error: " + e.getMessage() + "]";
            SwingUtilities.invokeLater(() -> updateChatArea(errorMessage));
            e.printStackTrace();
        } finally {
            SwingUtilities.invokeLater(() -> setGeneratingState(false));
            SwingUtilities.invokeLater(() -> updateChatArea("\n\n"));
        }
    }

    private void updateChatArea(String text) {
        chatArea.append(text);
        chatArea.setCaretPosition(chatArea.getDocument().getLength()); // Auto-scroll
    }

    private void setGeneratingState(boolean generating) {
        this.isGenerating = generating;
        promptField.setEnabled(!generating);
        sendButton.setEnabled(!generating);
        statusLabel.setText(generating ? "Generating..." : "Ready");
    }

    public static void main(String[] args) {
        // Run the GUI on the Event Dispatch Thread (EDT) for thread safety
        SwingUtilities.invokeLater(() -> {
            try {
                // Use a more modern look and feel if available
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception e) {
                e.printStackTrace();
            }
            new LlamaGuiJava().setVisible(true);
        });
    }
}
