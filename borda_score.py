import matplotlib.pyplot as plt
from collections import defaultdict

# Data ranking dari berbagai metode
rankings = {
    'skb_chi2': ['SrcBytes', 'TotBytes', 'SrcAddr', 'DstAddr', 'Sport', 'Dport', 'TotPkts', 'State', 'Dur', 'tcp', 'Dir', 'udp', 'icmp', 'igmp', 'rtp', 'rtcp', 'arp', 'ipv6-icmp', 'ipx/spx', 'ipv6', 'pim', 'udt', 'esp', 'rarp', 'unas', 'llc', 'gre', 'rsvp', 'ipnip'], 
    'skb_af': ['State', 'tcp', 'udp', 'Dir', 'Sport', 'DstAddr', 'Dport', 'SrcAddr', 'SrcBytes', 'Dur', 'icmp', 'TotBytes', 'TotPkts', 'igmp', 'rtp', 'rtcp', 'arp', 'ipv6-icmp', 'ipx/spx', 'ipv6', 'pim', 'udt', 'esp', 'rarp', 'unas', 'llc', 'gre', 'rsvp', 'ipnip'], 
    'skb_mi': ['SrcAddr', 'DstAddr', 'Dur', 'TotBytes', 'Dport', 'Sport', 'SrcBytes', 'State', 'TotPkts', 'Dir', 'udp', 'tcp', 'icmp', 'gre', 'llc', 'ipv6-icmp', 'pim', 'rtp', 'rtcp', 'rarp', 'igmp', 'arp', 'rsvp', 'udt', 'esp', 'ipv6', 'unas', 'ipnip', 'ipx/spx'], 
    'vt': ['TotBytes', 'SrcBytes', 'SrcAddr', 'DstAddr', 'Dport', 'Sport', 'TotPkts', 'Dur', 'State', 'Dir', 'udp', 'tcp', 'icmp', 'igmp', 'rtp', 'rtcp', 'arp', 'ipv6-icmp', 'ipx/spx', 'ipv6', 'pim', 'udt', 'esp', 'rarp', 'unas', 'llc', 'gre', 'rsvp', 'ipnip'], 
    'be': ['SrcAddr', 'Sport', 'Dport', 'tcp', 'SrcBytes', 'DstAddr', 'TotBytes', 'Dur', 'State', 'TotPkts', 'udp', 'Dir', 'rtp', 'rarp', 'ipx/spx', 'rtcp', 'igmp', 'ipnip', 'unas', 'gre', 'pim', 'ipv6', 'llc', 'esp', 'icmp', 'ipv6-icmp', 'arp', 'rsvp', 'udt'], 
    'rfe': ['SrcAddr', 'Dport', 'Sport', 'tcp', 'SrcBytes', 'DstAddr', 'TotBytes', 'Dur', 'State', 'TotPkts', 'Dir', 'udp', 'rtp', 'esp', 'rtcp', 'rarp', 'ipx/spx', 'ipnip', 'igmp', 'unas', 'llc', 'gre', 'pim', 'ipv6', 'icmp', 'ipv6-icmp', 'arp', 'udt', 'rsvp'], 
    'sfm_tb': ['SrcAddr', 'Sport', 'Dport', 'DstAddr', 'State', 'TotBytes', 'SrcBytes', 'Dur', 'TotPkts', 'tcp', 'udp', 'Dir', 'icmp', 'igmp', 'rtp', 'rtcp', 'arp', 'ipv6-icmp', 'ipx/spx', 'esp', 'ipv6', 'pim', 'llc', 'udt', 'unas', 'rarp', 'gre', 'rsvp', 'ipnip'], 
}

# Menghitung skor Borda untuk setiap fitur
borda_scores = defaultdict(int)
n_methods = len(rankings)

for ranking in rankings.values():
    for position, feature in enumerate(ranking):
        borda_scores[feature] += (len(ranking) - position)  # Skor dihitung berdasarkan posisi ranking

# Mengurutkan fitur berdasarkan skor tertinggi
sorted_features = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)

# Memisahkan fitur dan skor untuk plotting
features, scores = zip(*sorted_features)

# Visualisasi hasil dengan bar chart
plt.figure(figsize=(10, 6))
plt.barh(features[::-1], scores[::-1], color='skyblue')  # Dibalik agar ranking tertinggi di atas
# plt.xlabel("Skor Borda")
plt.xlabel("Borda score")
# plt.ylabel("Fitur")
plt.ylabel("Feature")
plt.title("Rank Aggregation using Borda Count")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
