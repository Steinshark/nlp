#!/usr/bin/python
import sys, os, operator, numpy, MySQLdb, json
import matplotlib.pyplot as plt

from db import entity
from db import session
from collections import defaultdict


'''
@author: anant bhardwaj
@date: Feb 12, 2013

script for preparing data in lenskit format
'''

entities = entity.Entity().entities
sessions = session.Session().sessions
connection = MySQLdb.connect(host="mysql.csail.mit.edu",
                     user="cobi",
                      passwd="su4Biha",
                      db="cobi")
nodes = {}
edges = defaultdict(dict)
likes = defaultdict(set)

scon = {}
papers_count = [];

def load_data():
	for p in entities:
		nodes[p] = {'title': entities[p]['title'], 'session': entities[p]['session'], 'award': entities[p]['award'], 'hm': entities[p]['hm']}	

	cursor = connection.cursor()
	cursor.execute("SELECT auth_no, likes, email1 FROM pcs_authors where likes!= 'NULL' and likes !='[]';")
	data = cursor.fetchall()
	for row in data:
		papers = json.loads(row[1])
		papers_count.append(len(papers))
		for p in papers:
			likes[p].add(row[0])


	for p1 in entities:
		for p2 in entities:
			edges[p1][p2] = -1
			if(p1 != p2):	
				common_likes = likes[p1].intersection(likes[p2])
				edges[p1][p2] = len(common_likes)

	for s in sessions:
		num_edges = 0
		papers = sessions[s]['submissions']
		for p1 in papers:
			for p2 in papers:
				try:
					if(p1 != p2 and edges[p1][p2] > 0):
						num_edges += 1
				except:
					pass
		if(len(sessions[s]['submissions']) > 0):
			scon[s] = float(num_edges)/len(sessions[s]['submissions'])
					


def main():
	load_data()
	nodesArray = []
	linksArray = []

	#print numpy.mean(scon.values()), numpy.std(scon.values()), numpy.median(scon.values()), numpy.min(scon.values()), numpy.max(scon.values())
	
	
	'''
	plt.hist([awards_count, non_awards_count], bins=100, histtype='bar', stacked=True, color=['yellow', 'green'],
                            label=['Award Papers', 'Non-Award Papers'])
	plt.title('Number of People Starred vs. Number of Papers')
	plt.xlabel('Number of People Starred')
	plt.ylabel('Number of Papers')
	
	plt.legend()
	plt.show()

	


	
	awards_count = []
	non_awards_count = []
	likes_count = [len(v) for k,v in likes.iteritems()]

	for k,v in likes.iteritems():
		if k in nodes and (nodes[k]['award'] or nodes[k]['hm']):
			awards_count.append(len(v))
		else:
			non_awards_count.append(len(v))

	
	#print numpy.mean(papers_count), numpy.std(papers_count), numpy.median(papers_count), numpy.min(papers_count), numpy.max(papers_count)
	print numpy.mean(likes_count), numpy.std(likes_count), numpy.median(likes_count), numpy.min(likes_count), numpy.max(likes_count)
	print numpy.mean(awards_count), numpy.std(awards_count), numpy.median(awards_count), numpy.min(awards_count), numpy.max(awards_count)
	
	plt.hist([awards_count, non_awards_count], bins=100, histtype='bar', stacked=True, color=['yellow', 'green'],
                            label=['Award Papers', 'Non-Award Papers'])
	plt.title('Number of People Starred vs. Number of Papers')
	plt.xlabel('Number of People Starred')
	plt.ylabel('Number of Papers')
	
	plt.legend()
	plt.show()
	
	plt.hist(papers_count, bins=20, color="cyan")
	plt.title('Number of Papers vs. Number of People')
	plt.xlabel('Number of Likes')
	plt.ylabel('Number of People')
	plt.show()
	'''
	
	
	
	k = 0

	for node in nodes:
		nodes[node]['id']= k
		nodesArray.append({'title' : nodes[node]['title'], 'session': nodes[node]['session'], 'weight': len(likes[node])})
		k = k+1
	
	edgesToRemove = set()
	'''
	for edge in edges:
		links = edges[edge]
		for l in links:
			weight = edges[edge][l]
			if(weight > 14):
				edgesToRemove.add(nodes[edge]['id'])
				edgesToRemove.add(nodes[l]['id'])
	'''
	

	for edge in edges:
		links = edges[edge]
		for l in links:
			weight = edges[edge][l]
			if(weight > 0 and (nodes[edge]['id'] not in edgesToRemove) and (nodes[l]['id'] not in edgesToRemove)):
				linksArray.append({'source' : nodes[edge]['id'], 'target' : nodes[l]['id'], 'weight': weight})
	
	p = open('data-all.json','w')
	p.write(json.dumps({"nodes": nodesArray, "links": linksArray}))


if __name__ == '__main__':
    main()
