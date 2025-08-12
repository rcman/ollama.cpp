import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.HashMap;
import java.util.Map;

public class ModelConfigGUI extends JFrame {
    private final Map<String, JComponent> inputFields = new HashMap<>();
    private JTextArea logArea;
    private boolean isModelRunning = false;
    private JButton startButton, stopButton;

    public ModelConfigGUI() {
        super("Model Configuration");
        setupUI();
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(900, 700);
        setLocationRelativeTo(null);
    }

    private void setupUI() {
        JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Create form panel with scrolling
        JPanel formPanel = createFormPanel();
        JScrollPane scrollPane = new JScrollPane(formPanel);
        scrollPane.setBorder(BorderFactory.createTitledBorder("Model Parameters"));

        // Create log panel
        JPanel logPanel = new JPanel(new BorderLayout());
        logArea = new JTextArea(8, 80);
        logArea.setEditable(false);
        logPanel.setBorder(BorderFactory.createTitledBorder("Log"));
        logPanel.add(new JScrollPane(logArea), BorderLayout.CENTER);

        // Create button panel
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 10));
        startButton = new JButton("Start Model");
        stopButton = new JButton("Stop Model");
        stopButton.setEnabled(false);
        
        startButton.addActionListener(this::startModel);
        stopButton.addActionListener(e -> stopModel());
        
        buttonPanel.add(startButton);
        buttonPanel.add(stopButton);

        // Assemble main panel
        mainPanel.add(scrollPane, BorderLayout.CENTER);
        mainPanel.add(logPanel, BorderLayout.SOUTH);
        mainPanel.add(buttonPanel, BorderLayout.NORTH);

        add(mainPanel);
    }

    private JPanel createFormPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        
        // Add parameter groups
        panel.add(createCoreParamsPanel());
        panel.add(createGPUPanel());
        panel.add(createThreadPanel());
        panel.add(createRoPEPanel());
        panel.add(createYaRNGPanel());
        panel.add(createCachePanel());
        panel.add(createHFPanel());
        panel.add(createMiscPanel());
        
        return panel;
    }

    private JPanel createCoreParamsPanel() {
        JPanel panel = new JPanel(new GridLayout(0, 2, 5, 5));
        panel.setBorder(new TitledBorder("Core Parameters"));
        
        addField(panel, "model", "Model path", String.class, "");
        addField(panel, "model_alias", "Model alias", String.class, "");
        addField(panel, "n_ctx", "Context size", int.class, "2048");
        addField(panel, "n_batch", "Batch size", int.class, "512");
        addField(panel, "n_ubatch", "Physical batch size", int.class, "512");
        addField(panel, "seed", "Random seed (-1=random)", int.class, "-1");
        addField(panel, "vocab_only", "Return only vocabulary", boolean.class, "false");
        addField(panel, "logits_all", "Return logits", boolean.class, "true");
        addField(panel, "embedding", "Use embeddings", boolean.class, "false");
        addField(panel, "last_n_tokens_size", "Repeat penalty tokens", int.class, "64");
        addField(panel, "verbose", "Print debug info", boolean.class, "true");
        
        return panel;
    }

    private JPanel createGPUPanel() {
        JPanel panel = new JPanel(new GridLayout(0, 2, 5, 5));
        panel.setBorder(new TitledBorder("GPU Configuration"));
        
        addField(panel, "n_gpu_layers", "GPU Layers (-1=all)", int.class, "0");
        addField(panel, "main_gpu", "Main GPU", int.class, "0");
        addField(panel, "tensor_split", "GPU Split proportions", String.class, "");
        addField(panel, "split_mode", "Split mode", int.class, "0");
        addField(panel, "offload_kqv", "Offload KQV to GPU", boolean.class, "true");
        addField(panel, "flash_attn", "Use flash attention", boolean.class, "false");
        
        return panel;
    }

    private JPanel createThreadPanel() {
        JPanel panel = new JPanel(new GridLayout(0, 2, 5, 5));
        panel.setBorder(new TitledBorder("Thread Configuration"));
        
        int defaultThreads = Math.max(Runtime.getRuntime().availableProcessors() / 2, 1);
        int defaultBatchThreads = Math.max(Runtime.getRuntime().availableProcessors(), 1);
        
        addField(panel, "n_threads", "Threads (-1=max)", int.class, String.valueOf(defaultThreads));
        addField(panel, "n_threads_batch", "Batch threads (-1=max)", int.class, String.valueOf(defaultBatchThreads));
        
        return panel;
    }

    private JPanel createRoPEPanel() {
        JPanel panel = new JPanel(new GridLayout(0, 2, 5, 5));
        panel.setBorder(new TitledBorder("RoPE Configuration"));
        
        addField(panel, "rope_scaling_type", "Scaling type", int.class, "0");
        addField(panel, "rope_freq_base", "Base frequency", double.class, "0.0");
        addField(panel, "rope_freq_scale", "Frequency scale", double.class, "0.0");
        
        return panel;
    }

    private JPanel createYaRNGPanel() {
        JPanel panel = new JPanel(new GridLayout(0, 2, 5, 5));
        panel.setBorder(new TitledBorder("YaRN Configuration"));
        
        addField(panel, "yarn_ext_factor", "Extrapolation factor", double.class, "-1.0");
        addField(panel, "yarn_attn_factor", "Attention factor", double.class, "1.0");
        addField(panel, "yarn_beta_fast", "Beta fast", double.class, "32.0");
        addField(panel, "yarn_beta_slow", "Beta slow", double.class, "1.0");
        addField(panel, "yarn_orig_ctx", "Original context", int.class, "0");
        
        return panel;
    }

    private JPanel createCachePanel() {
        JPanel panel = new JPanel(new GridLayout(0, 2, 5, 5));
        panel.setBorder(new TitledBorder("Cache Configuration"));
        
        addField(panel, "cache", "Use cache", boolean.class, "false");
        addField(panel, "cache_type", "Cache type", String.class, "ram");
        addField(panel, "cache_size", "Cache size (bytes)", long.class, String.valueOf(2 << 30));
        
        return panel;
    }

    private JPanel createHFPanel() {
        JPanel panel = new JPanel(new GridLayout(0, 2, 5, 5));
        panel.setBorder(new TitledBorder("HuggingFace Configuration"));
        
        addField(panel, "hf_tokenizer_config_path", "Tokenizer config path", String.class, "");
        addField(panel, "hf_pretrained_model_name_or_path", "Pretrained model path", String.class, "");
        addField(panel, "hf_model_repo_id", "Model repo ID", String.class, "");
        
        return panel;
    }

    private JPanel createMiscPanel() {
        JPanel panel = new JPanel(new GridLayout(0, 2, 5, 5));
        panel.setBorder(new TitledBorder("Miscellaneous Configuration"));
        
        addField(panel, "use_mmap", "Use mmap", boolean.class, "true");
        addField(panel, "use_mlock", "Use mlock", boolean.class, "false");
        addField(panel, "kv_overrides", "KV overrides", String.class, "");
        addField(panel, "rpc_servers", "RPC servers", String.class, "");
        addField(panel, "lora_base", "LoRA base model", String.class, "");
        addField(panel, "lora_path", "LoRA path", String.class, "");
        addField(panel, "numa", "NUMA support", boolean.class, "false");
        addField(panel, "chat_format", "Chat format", String.class, "");
        addField(panel, "clip_model_path", "CLIP model path", String.class, "");
        addField(panel, "mul_mat_q", "Use mul_mat_q kernels", boolean.class, "true");
        addField(panel, "draft_model", "Draft model method", String.class, "");
        addField(panel, "draft_model_num_pred_tokens", "Prediction tokens", int.class, "10");
        addField(panel, "type_k", "Key cache type", String.class, "");
        addField(panel, "type_v", "Value cache type", String.class, "");
        
        return panel;
    }

    private void addField(JPanel panel, String name, String label, Class<?> type, String defaultValue) {
        JLabel jLabel = new JLabel(label + ":");
        JComponent field;
        
        if (type == boolean.class) {
            JCheckBox checkBox = new JCheckBox();
            checkBox.setSelected(Boolean.parseBoolean(defaultValue));
            field = checkBox;
        } else if (type == int.class || type == long.class || type == double.class) {
            field = new JTextField(defaultValue, 10);
        } else {
            field = new JTextField(defaultValue, 20);
        }
        
        panel.add(jLabel);
        panel.add(field);
        inputFields.put(name, field);
    }

    private void startModel(ActionEvent e) {
        if (isModelRunning) {
            log("Model is already running");
            return;
        }
        
        Map<String, Object> config = collectConfig();
        log("Starting model with configuration:");
        config.forEach((k, v) -> log(k + " = " + v));
        
        isModelRunning = true;
        startButton.setEnabled(false);
        stopButton.setEnabled(true);
        log("Model started successfully");
    }

    private void stopModel() {
        if (!isModelRunning) {
            log("No model is currently running");
            return;
        }
        
        isModelRunning = false;
        startButton.setEnabled(true);
        stopButton.setEnabled(false);
        log("Model stopped");
    }

    private Map<String, Object> collectConfig() {
        Map<String, Object> config = new HashMap<>();
        
        inputFields.forEach((name, field) -> {
            if (field instanceof JCheckBox) {
                config.put(name, ((JCheckBox) field).isSelected());
            } else if (field instanceof JTextField) {
                String text = ((JTextField) field).getText();
                try {
                    if (name.equals("n_gpu_layers")) config.put(name, parseInt(text));
                    else if (name.equals("n_ctx")) config.put(name, parseInt(text));
                    else if (name.equals("n_batch")) config.put(name, parseInt(text));
                    else if (name.equals("n_ubatch")) config.put(name, parseInt(text));
                    else if (name.equals("seed")) config.put(name, parseInt(text));
                    else if (name.equals("last_n_tokens_size")) config.put(name, parseInt(text));
                    else if (name.equals("rope_scaling_type")) config.put(name, parseInt(text));
                    else if (name.equals("rope_freq_base")) config.put(name, parseDouble(text));
                    else if (name.equals("rope_freq_scale")) config.put(name, parseDouble(text));
                    else if (name.equals("yarn_ext_factor")) config.put(name, parseDouble(text));
                    else if (name.equals("yarn_attn_factor")) config.put(name, parseDouble(text));
                    else if (name.equals("yarn_beta_fast")) config.put(name, parseDouble(text));
                    else if (name.equals("yarn_beta_slow")) config.put(name, parseDouble(text));
                    else if (name.equals("yarn_orig_ctx")) config.put(name, parseInt(text));
                    else if (name.equals("n_threads")) config.put(name, parseInt(text));
                    else if (name.equals("n_threads_batch")) config.put(name, parseInt(text));
                    else if (name.equals("main_gpu")) config.put(name, parseInt(text));
                    else if (name.equals("draft_model_num_pred_tokens")) config.put(name, parseInt(text));
                    else if (name.equals("cache_size")) config.put(name, parseLong(text));
                    else config.put(name, text);
                } catch (NumberFormatException ex) {
                    config.put(name, text); // Preserve as string if invalid number
                }
            }
        });
        
        return config;
    }

    private Integer parseInt(String text) {
        return text.isEmpty() ? 0 : Integer.parseInt(text);
    }

    private Double parseDouble(String text) {
        return text.isEmpty() ? 0.0 : Double.parseDouble(text);
    }

    private Long parseLong(String text) {
        return text.isEmpty() ? 0L : Long.parseLong(text);
    }

    private void log(String message) {
        logArea.append(message + "\n");
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            ModelConfigGUI gui = new ModelConfigGUI();
            gui.setVisible(true);
        });
    }
}