from dataclasses import dataclass

@dataclass
class Bid():
    bidder: str
    bid: int

class MarketEnvironment():
    def __init__(self, num_of_agents):
        self.agents = num_of_agents
        self.market = {}
        
    def reset(self):
        self.market = {}
        
    def get_market(self):
        return self.market
    
    def place_job(self, job):
        self.market[job] = {}
    
    def get_bids(self, job):
        return self.market[job]
    
    def place_bid(self, job, bidder, bid):
        if not job in self.market:
            raise ValueError(f"Job {job} does not exist on the market!")
        
        bid = Bid(bidder, bid)
        self.market[job].append(bid)
        self.market[job].sort(key=lambda b: b.bid, reverse=True)
    
    def determine_winner(self, job):
        bids = self.market.get(job, [])
        if not bids:
            return None
        return self.market[job][0]