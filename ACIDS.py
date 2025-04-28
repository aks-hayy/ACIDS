#!/usr/bin/env python3
import numpy as np
import torch
from scapy.all import *
import tkinter as tk
from tkinter import ttk
from threading import Thread, Event
from queue import Queue
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN  # More robust than KMeans
from sklearn.manifold import TSNE
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class EnhancedVerificationApp:
    def __init__(self, master, packet_queue):
        self.master = master
        self.packet_queue = packet_queue
        self.verification_queue = Queue()
        self.current_groups = []
        self.stats = {'fp': 0, 'real': 0, 'threshold': 0.05}
        self.fp_history = []
        
        # GUI Setup
        master.title("Enhanced Anomaly Verification")
        master.geometry("1200x800")
        
        # Main container
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Cluster display
        self.cluster_frame = ttk.LabelFrame(main_frame, text="Anomaly Groups", padding="10")
        self.cluster_frame.pack(fill=tk.BOTH, expand=True)
        
        # Visualization frame
        self.viz_frame = ttk.Frame(main_frame)
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Stats panel
        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_label = ttk.Label(stats_frame, 
                                   text=f"Stats: 0 FPs | 0 Real | Threshold: {self.stats['threshold']:.4f}")
        self.stats_label.pack(side=tk.LEFT)
        
        ttk.Button(stats_frame, text="Save & Close", 
                  command=self.save_and_close).pack(side=tk.RIGHT)
        
        # Initialize
        self.update_display()
        
    def extract_features_for_clustering(self, anomalies):
        """Create robust feature vectors for clustering"""
        features = []
        for a in anomalies:
            pkt = a['packet']
            features.append([
                a['error'],  # Reconstruction error
                hash(pkt[IP].src) % 1000,  # Source IP (hashed)
                pkt.dport if pkt.haslayer(TCP) or pkt.haslayer(UDP) else 0,  # Dest port
                len(pkt),  # Packet size
                pkt[IP].ttl if pkt.haslayer(IP) else 64,  # TTL
                hash(pkt[IP].dst) % 1000  # Dest IP (hashed)
            ])
        return StandardScaler().fit_transform(features)  # Normalize features
    
    def cluster_anomalies(self, anomalies):
        """Improved clustering using DBSCAN"""
        if len(anomalies) < 10:
            return [{'items': anomalies, 'key': 'all'}]
            
        X = self.extract_features_for_clustering(anomalies)
        
        # Adaptive DBSCAN parameters
        eps = 0.5 + (1.0 / len(anomalies))  # Auto-adjust based on sample size
        dbscan = DBSCAN(eps=eps, min_samples=3).fit(X)
        
        clusters = defaultdict(list)
        for i, label in enumerate(dbscan.labels_):
            clusters[label].append(anomalies[i])
            
        # Sort clusters by size
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Visualize clusters
        self.visualize_clusters(X, dbscan.labels_)
        
        return [{'items': v, 'key': f"Group {k+1}"} for k, (_, v) in enumerate(sorted_clusters[:5])]  # Top 5 clusters
    
    def visualize_clusters(self, X, labels):
        """Visualize clusters using t-SNE"""
        self.ax.clear()
        
        # Reduce dimensions for visualization
        if X.shape[1] > 2:
            X_embedded = TSNE(n_components=2, perplexity=min(30, len(X)-1)).fit_transform(X)
        else:
            X_embedded = X
            
        # Plot clusters
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'k'  # Noise shown as black
            
            class_member_mask = (labels == k)
            xy = X_embedded[class_member_mask]
            self.ax.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'Cluster {k}')
        
        self.ax.set_title('t-SNE Cluster Visualization')
        self.ax.legend()
        self.canvas.draw()
    
    def update_display(self):
        # Check for new anomalies
        try:
            new_anomalies = self.packet_queue.get_nowait()
            self.current_groups = self.cluster_anomalies(new_anomalies)
            self.update_group_display()
        except:
            pass
            
        self.master.after(1000, self.update_display)
    
    def update_group_display(self):
        # Clear current display
        for widget in self.cluster_frame.winfo_children():
            widget.destroy()
            
        # Display each cluster group
        for group in self.current_groups:
            group_frame = ttk.Frame(self.cluster_frame, relief=tk.GROOVE, borderwidth=1)
            group_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Group header
            header_frame = ttk.Frame(group_frame)
            header_frame.pack(fill=tk.X)
            
            ttk.Label(header_frame, text=group['key'], 
                     font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT)
            
            # Group stats
            avg_error = np.mean([a['error'] for a in group['items']])
            sample = group['items'][0]
            info_text = (f"{len(group['items'])} packets | "
                        f"Avg Error: {avg_error:.2f} | "
                        f"{sample['packet'][IP].src} → {sample['packet'][IP].dst} "
                        f"Port: {sample['packet'].dport if sample['packet'].haslayer(TCP) else 'N/A'}")
            
            ttk.Label(header_frame, text=info_text).pack(side=tk.LEFT, padx=10)
            
            # Action buttons
            btn_frame = ttk.Frame(group_frame)
            btn_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(btn_frame, text="Mark ALL as FP",
                      command=lambda g=group: self.mark_group(g, 'fp'),
                      style='danger.TButton').pack(side=tk.LEFT, padx=5)
                      
            ttk.Button(btn_frame, text="Mark ALL as Real",
                      command=lambda g=group: self.mark_group(g, 'real'),
                      style='success.TButton').pack(side=tk.LEFT)
                      
            ttk.Button(btn_frame, text="Inspect Sample",
                      command=lambda g=group: self.show_sample(g)).pack(side=tk.RIGHT)
    
    def mark_group(self, group, verdict):
        for item in group['items']:
            item['verdict'] = verdict
            self.stats[verdict] += 1
            if verdict == 'fp':
                self.fp_history.append(item['error'])
        
        # Update threshold based on FP history
        if verdict == 'fp' and len(self.fp_history) >= 10:
            new_threshold = np.percentile(self.fp_history, 95) * 1.1  # 95th percentile + 10% margin
            self.stats['threshold'] = max(self.stats['threshold'], new_threshold)
        
        self.update_stats()
        self.current_groups.remove(group)
        self.update_group_display()
    
    def update_stats(self):
        self.stats_label.config(
            text=f"Stats: {self.stats['fp']} FPs | {self.stats['real']} Real | Threshold: {self.stats['threshold']:.4f}"
        )
    
    def show_sample(self, group):
        sample = group['items'][0]
        detail_win = tk.Toplevel(self.master)
        detail_win.title(f"Sample - {group['key']}")
        
        text = tk.Text(detail_win, wrap=tk.WORD, width=100, height=15)
        text.pack(fill=tk.BOTH, expand=True)
        
        pkt = sample['packet']
        text.insert(tk.END, f"Error: {sample['error']:.4f}\n")
        text.insert(tk.END, f"Source: {pkt[IP].src}:{pkt.sport if pkt.haslayer(TCP) or pkt.haslayer(UDP) else ''}\n")
        text.insert(tk.END, f"Destination: {pkt[IP].dst}:{pkt.dport if pkt.haslayer(TCP) or pkt.haslayer(UDP) else ''}\n")
        text.insert(tk.END, f"Protocol: {pkt.sprintf('%IP.proto%')}\n")
        text.insert(tk.END, f"Size: {len(pkt)} bytes\n")
        text.insert(tk.END, f"TTL: {pkt[IP].ttl if pkt.haslayer(IP) else ''}\n\n")
        text.insert(tk.END, "Packet Summary:\n")
        text.insert(tk.END, pkt.summary())
        
        if pkt.haslayer(Raw):
            text.insert(tk.END, f"\n\nPayload (first 100 bytes):\n{pkt[Raw].load[:100]}")
        
        text.config(state=tk.DISABLED)
    
    def save_and_close(self):
        verified = []
        for group in self.current_groups:
            verified.extend(group['items'])
            
        self.verification_queue.put({
            'batch': verified,
            'fps': self.stats['fp'],
            'reals': self.stats['real'],
            'new_threshold': self.stats['threshold']
        })
        self.master.destroy()


class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x        

def safe_preprocess(features):
    """Robust preprocessing with NaN handling"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Protocol encoding with fallback
        protocol_map = {'tcp': 0, 'udp': 1, 'icmp': 2, 'other': 3}
        df['protocol_type'] = df['protocol_type'].map(protocol_map).fillna(3)
        
        # Service encoding with fallback  
        service_map = {'http': 0, 'dns': 1, 'other': 2}
        df['service'] = df['service'].map(service_map).fillna(2)
        
        # Flag encoding with fallback
        flag_map = {'SF': 0, 'S0': 1, 'REJ': 2, 'OTH': 3}
        df['flag'] = df['flag'].map(flag_map).fillna(3)
        
        # Fill any remaining NaNs with 0
        df = df.fillna(0)
        
        # Ensure we have all expected columns
        expected_cols = ['duration', 'protocol_type', 'service', 'flag', 
                        'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                        'urgent', 'hot', 'num_failed_logins', 'logged_in',
                        'num_compromised', 'root_shell', 'su_attempted',
                        'num_root', 'num_file_creations', 'num_shells',
                        'num_access_files', 'num_outbound_cmds', 'is_host_login',
                        'is_guest_login', 'count', 'srv_count', 'serror_rate',
                        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                        'dst_host_srv_rerror_rate']
        
        # Add missing columns with default values
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Scale features (use your trained scaler in production)
        numerical_cols = [c for c in expected_cols if c not in ['protocol_type', 'service', 'flag']]
        df[numerical_cols] = MinMaxScaler().fit_transform(df[numerical_cols])
        
        # Convert to tensor and reshape
        X = df[expected_cols].values.astype(np.float32)
        X = np.nan_to_num(X)  # Convert any remaining NaNs to 0
        
        # Pad and reshape for CNN
        X = np.pad(X, ((0, 0), (0, 64 - X.shape[1])))
        X = X[:, :64].reshape(-1, 1, 8, 8)
        
        return torch.from_numpy(X)
    
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def extract_features(packet):
    features = {
        'duration': 0,
        'protocol_type': 0,  # Will be encoded later
        'service': 0,       # Will be encoded later
        'flag': 0,          # Will be encoded later
        'src_bytes': len(packet) if IP in packet else 0,
        'dst_bytes': 0,     # Can't determine for single packet
        'land': 1 if packet.haslayer(IP) and packet[IP].src == packet[IP].dst else 0,
        'wrong_fragment': 0,
        'urgent': 1 if packet.haslayer(TCP) and packet[TCP].flags & 0x20 else 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 0,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 1,  # Single packet
        'srv_count': 0,
        'serror_rate': 0,
        'srv_serror_rate': 0,
        'rerror_rate': 0,
        'srv_rerror_rate': 0,
        'same_srv_rate': 0,
        'diff_srv_rate': 0,
        'srv_diff_host_rate': 0,
        'dst_host_count': 0,
        'dst_host_srv_count': 0,
        'dst_host_same_srv_rate': 0,
        'dst_host_diff_srv_rate': 0,
        'dst_host_same_src_port_rate': 0,
        'dst_host_srv_diff_host_rate': 0,
        'dst_host_serror_rate': 0,
        'dst_host_srv_serror_rate': 0,
        'dst_host_rerror_rate': 0,
        'dst_host_srv_rerror_rate': 0,
    }

    # Protocol type encoding
    if packet.haslayer(IP):
        if packet.haslayer(TCP):
            features['protocol_type'] = 'tcp'
        elif packet.haslayer(UDP):
            features['protocol_type'] = 'udp'
        elif packet.haslayer(ICMP):
            features['protocol_type'] = 'icmp'
        else:
            features['protocol_type'] = 'other'

    # Service detection (simplified)
    if packet.haslayer(TCP):
        features['service'] = 'http' if packet[TCP].dport == 80 else 'other'
    elif packet.haslayer(UDP):
        features['service'] = 'dns' if packet[UDP].dport == 53 else 'other'
    else:
        features['service'] = 'other'

    # TCP flags
    if packet.haslayer(TCP):
        flags = packet[TCP].flags
        if flags & 0x02:  # SYN
            features['flag'] = 'SF' if flags & 0x10 else 'S0'
        elif flags & 0x10:  # ACK
            features['flag'] = 'SF'
        else:
            features['flag'] = 'REJ'
    else:
        features['flag'] = 'OTH'

    return features

    
class NetworkMonitor:
    def __init__(self):
        self.packet_queue = Queue()
        self.verification_queue = Queue()
        self.anomaly_buffer = []
        self.threshold = 0.05
        self.fp_history = []
        
        # Load model
        self.model = CNNAutoencoder()
        self.model.load_state_dict(torch.load("cnn_autoencoder_nslkdd.pth"))
        self.model.eval()
        
    def packet_handler(self, packet):
        try:
            if not packet.haslayer(IP):
                return
                
            features = extract_features(packet)
            X = safe_preprocess(features)
            
            if X is None:
                return
                
            with torch.no_grad():
                reconstructed = self.model(X)
                error = torch.nn.functional.mse_loss(reconstructed, X).item()
                
            if error > self.threshold:
                self.anomaly_buffer.append({
                    'packet': packet,
                    'error': error,
                    'timestamp': time.time()
                })
                
                # Trigger verification every 30 anomalies (smaller batches)
                if len(self.anomaly_buffer) >= 30:
                    self.packet_queue.put(self.anomaly_buffer)
                    self.anomaly_buffer = []
                    
        except Exception as e:
            print(f"Packet processing error: {e}")
            
    def adjust_threshold(self, verification_results):
        if 'new_threshold' in verification_results:
            new_threshold = verification_results['new_threshold']
            if new_threshold > self.threshold:
                self.threshold = new_threshold
                print(f"Threshold updated to: {self.threshold:.4f}")
    
    def start_monitoring(self):
        sniff_thread = Thread(target=lambda: sniff(prn=self.packet_handler, store=0))
        sniff_thread.daemon = True
        sniff_thread.start()
        
        while True:
            # Process verification results
            try:
                results = self.verification_queue.get_nowait()
                self.adjust_threshold(results)
            except:
                pass
                
            time.sleep(0.1)


if __name__ == "__main__":
    # Create monitoring system
    monitor = NetworkMonitor()
    
    # Start GUI with styles
    root = tk.Tk()
    style = ttk.Style()
    style.configure('danger.TButton', foreground='white', background='#dc3545')
    style.configure('success.TButton', foreground='white', background='#28a745')
    style.map('danger.TButton', background=[('active', '#c82333')])
    style.map('success.TButton', background=[('active', '#218838')])
    
    app = EnhancedVerificationApp(root, monitor.packet_queue)
    
    # Start monitoring in background
    monitor_thread = Thread(target=monitor.start_monitoring)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    root.mainloop()
    
    # After GUI closes
    monitor_thread.join()